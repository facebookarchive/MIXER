--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Marc'Aurelio Ranzato <ranzato@fb.com>
--          Sumit Chopra <spchopra@fb.com>
--          Michael Auli <michaelauli@fb.com>
--          Wojciech Zaremba <zaremba@cs.nyu.edu>
--
require('torch')
require('sys')
require('nn')
require('xlua')
local utils = paths.dofile('util.lua')
paths.dofile('ClassNLLCriterionWeighted.lua')
paths.dofile('LinearNoBackpropInput.lua')

-- Recurrent neural network supporting both cross-entropy and REINFORCE loss.
local Mixer = torch.class("Mixer")

function Mixer:__init(config, net, criterion, double)
    self.type = double and 'torch.DoubleTensor' or 'torch.CudaTensor'
    self.net = net:clone()
    self.criterion = criterion -- Reinforce Criterion
    -- build the predictors of the cumulative reward
    self.cum_reward_predictors = {}
    self.nlls = {} -- xent losses at each time step
    self.wxent = 0.5
    self.eosidx = config.eosIndex
    self.dict_size = config.n_tokens
    self.padidx = config.paddingIndex
    local weights = torch.ones(config.n_tokens):type(self.type)
    weights[self.padidx] = 0 -- always skip padding token
    assert(self.wxent >= 0 and self.wxent <= 1)
    self.criterion:set_weight(1 - self.wxent)
    for tt = 1, config.bptt do
        self.cum_reward_predictors[tt] =
            nn.LinearNoBackpropInput(2 * config.n_hidden, 1)
        self.cum_reward_predictors[tt].bias:fill(0.01)
        self.cum_reward_predictors[tt].weight:fill(0)
        self.nlls[tt] = nn.ClassNLLCriterionWeighted(
            self.wxent, weights, true)
    end
    if not double then
        self.net:cuda()
        self.criterion:cuda()
        for tt = 1, config.bptt do
            self.cum_reward_predictors[tt] =
                self.cum_reward_predictors[tt]:cuda()
            self.nlls[tt] = self.nlls[tt]:cuda()
        end
    end
    self.param, self.grad_param = self.net:getParameters()
    self.param_crp = {}
    self.grad_param_crp = {}
    for tt = 1, config.bptt do
        self.param_crp[tt], self.grad_param_crp[tt] =
            self.cum_reward_predictors[tt]:getParameters()
    end
    self.initial_val = 0
    self.initial_state_dim = config.size_hid_layers
    -- we are going to brpop gradients from criterion only for the last
    -- config.bptt steps, yet bprop through the rest of the RNN for
    -- config.bptt + config.nrstepsinit steps
    self.bptt = config.bptt -- tot nr unrollng steps
    self.nrstepsinit = 1
    self.batch_size = config.batch_size
    self.hiddens = self:_init_state(config.initial_val)
    self.gradient_hiddens = self:_init_state(0)
    self.pred = {} -- stores {sampled_word, logprob, output_encoder}
    self.pred_dx = {}
    self.pred_rf = {} -- stores {sampled_word, cumulative reward prediction}
    self.pred_rf_dx = {}
    for tt = 1, self.bptt do
        self.pred_rf[tt] = {}
        self.pred_dx[tt] = {}
        self.pred_dx[tt][2] =
            torch.zeros(self.batch_size, config.n_tokens):type(self.type)
    end
    self.inputs = {}
    self.labels_xent = {}
    self.labels = {}
    self.clip_function = utils.scale_clip
    self.clip_param_val = config.grad_param_clip
    -- unroll the network over time
    self:_unroll()
    self:reset()
    self.tot_reward = 0
    self.num_samples_rf = 0
    self.tot_cumreward_pred_error = 0
end

-- Initialize the hidden states.
function Mixer:_init_state(val)
    local new_state = {}
    for tt = 0, self.bptt do
        new_state[tt] = {}
        for hh = 1, #self.initial_state_dim do
            new_state[tt][hh] =
                torch.Tensor(self.batch_size, self.initial_state_dim[hh]):type(
                    self.type)
            new_state[tt][hh]:fill((tt == 0) and val or 0)
        end
    end
    return new_state
end

-- Reset network (gradients of parameters and hidden states)
function Mixer:reset()
   self.i_input = 0
   self.grad_param:zero()
   -- reset hidden states
   for hh = 1, #self.hiddens[0] do
       self.hiddens[0][hh]:fill(self.initial_val)
   end
end

-- load the previously saved model. User has the option of
-- specifying the batch size, which will be used to create and
-- initialize the new hidden states.
function Mixer:load(mfile)
    print('[[ loading previously trained model ' .. mfile .. ' ]] ')
    local stored_model = torch.load(mfile)
    self.param, self.grad_param = self.net:getParameters()
    local stored_param, stored_grad_param = stored_model.net:getParameters()
    self.param:copy(stored_param)
    self.grad_param:copy(stored_grad_param)
    self.initial_val = stored_model.initial_val
    self.initial_state_dim = stored_model.initial_state_dim
    self.nrstepsinit = nrstepsinit or stored_model.nrstepsinit
    self.clip_param_val = stored_model.clip_param_val
    self.hiddens = self:_init_state(self.initial_val)
    self.gradient_hiddens = self:_init_state(0)
    self:_unroll()
    self:reset()
end

function Mixer:save(fname)
    local save_model = {}
    save_model.net = self.net
    save_model.cum_reward_predictors = self.cum_reward_predictors
    save_model.initial_val = self.initial_val
    save_model.initial_state_dim = self.initial_state_dim
    save_model.bptt = self.bptt
    save_model.nrstepsinit = self.nrstepsinit
    save_model.batch_size = self.batch_size
    save_model.clip_param_val = self.clip_param_val
    -- save the model
    torch.save(fname, save_model)
end

-- Actual unfolding of the RNN through time.
function Mixer:_unroll()
    self.unrolled_nets = {}
    for tt = 1, self.bptt do
        self.unrolled_nets[tt] = self.net:clone('weight', 'bias',
                                                'gradWeight', 'gradBias')
    end
end

function Mixer:reset_rf_vars()
   self.num_samples_rf = 0
   self.tot_cumreward_pred_error = 0
   self.tot_reward = 0
end

function Mixer:reset()
    self.i_input = 0
    self.grad_param:zero()
    -- reset hidden states
    for hh = 1, #self.hiddens[0] do
        self.hiddens[0][hh]:fill(self.initial_val)
    end
end

function Mixer:get_rf_vars()
   return self.num_samples_rf, self.tot_cumreward_pred_error,
          self.tot_reward
end

function Mixer:set_xent_weight(ww)
   assert(ww >= 0 and ww <= 1)
   self.wxent = ww
   for tt = 1, self.bptt do
      self.nlls[tt].globalWeight = ww
   end
   self.criterion:set_weight(1 - ww)
end

function Mixer:set_nrstepsinit(nr)
   self.nrstepsinit = nr
   self.criterion:set_skips(nr)
end

-- Perform one step of pure gradient descent.
function Mixer:_updateParams(learning_rate)
    self.param:add(- learning_rate, self.grad_param)
    for tt = 1, self.bptt do
        self.param_crp[tt]:add(- learning_rate, self.grad_param_crp[tt])
    end
end

function Mixer:overwrite_prediction(prev, curr)
    -- when previous token was eos or PAD,
    -- PAD is produced deterministically.
    for ss = 1, prev:size(1) do
        if prev[ss][1] == self.eosidx or prev[ss][1] == self.padidx then
            curr[ss][1] = self.padidx
        end
    end
end

-- This function performs a forward pass through one time step.
-- Every bptt steps it also performs bprop through all the time steps.
function Mixer:train_one_batch(input, label, learning_rate)
   -- FPROP
   local loss_xe = 0 -- XENT
   local loss_rf = 0 -- REINFORCE
   local num_samples = 0 -- because of padding this maybe less than mbsz
   local num_samples_rf = 0
   local step = self.i_input % self.bptt
   self.inputs[step] = {}
   for k, v in pairs(input) do
       self.inputs[step][k] = v:clone()
   end
   if step >= self.nrstepsinit then
       self.inputs[step][1] = self.pred[step][1]:squeeze()
   end
   self.labels[step + 1] = label
   -- pred stores: sampled word, logprob scores, output of encoder
   -- fprop 1 step through RNN
   self.pred[step + 1], self.hiddens[step + 1] =
      unpack(self.unrolled_nets[step + 1]:forward(
                 {self.inputs[step], self.hiddens[step]}))
   -- and through cumulative reward predictor at that step
   self.pred_rf[step + 1][2] = self.cum_reward_predictors[step + 1]:forward(
      self.pred[step + 1][3])
   if step + 1 < self.nrstepsinit then
       -- overwrite prediction with ground truth label
       self.pred[step + 1][1]:copy(label)
   else
       -- skip first step since there is no history to carry over
       if step > 0 then
           self:overwrite_prediction(self.pred[step][1], self.pred[step + 1][1])
       end
   end
   self.pred_rf[step + 1][1] = self.pred[step + 1][1]:squeeze()
   if step < self.nrstepsinit then
       -- and through cross entropy loss for next symbol
       loss_xe, num_samples = self.nlls[step + 1]:forward(
           self.pred[step + 1][2], self.labels[step + 1])
   else
      loss_xe = 0
      num_samples = 0
   end
   -- Every bptt steps, do bprop.
   if step + 1 == self.bptt then
      -- reinforce criterion operates on the whole sequence
      loss_rf, num_samples_rf =
         self.criterion:forward(self.pred_rf, self.labels)
      -- BPROP
      self.pred_rf_dx = self.criterion:backward(self.pred_rf, self.labels)
      self.tot_reward = self.tot_reward -
         (self.criterion.sizeAverage and loss_rf * num_samples_rf or loss_rf)
      self.tot_cumreward_pred_error = self.tot_cumreward_pred_error +
         self.criterion.gradInput[self.nrstepsinit][1]:norm()
      self.num_samples_rf = self.num_samples_rf + num_samples_rf
      self.grad_param:zero()
      for tt = self.bptt, 1, -1 do
         self.grad_param_crp[tt]:zero()
         if tt <= self.nrstepsinit then
             self.pred_dx[tt][2] = self.nlls[tt]:backward(self.pred[tt][2],
                                                          self.labels[tt])
         else
            self.pred_dx[tt][2]:fill(0)
         end
         self.pred_dx[tt][1] = self.pred_rf_dx[tt][1]:view(self.batch_size, 1)
         self.pred_dx[tt][3] = self.cum_reward_predictors[tt]:backward(
            self.pred[tt][3], self.pred_rf_dx[tt][2])
         self.gradient_hiddens[tt - 1]  = self.unrolled_nets[tt]:backward(
             {self.inputs[tt - 1], self.hiddens[tt - 1]},
             {self.pred_dx[tt], self.gradient_hiddens[tt]})[2]
      end
      -- Update parameters
      if self.clip_param_val then
          self.clip_function(self.grad_param, self.clip_param_val)
      end
      self:_updateParams(learning_rate)
   end
   self.i_input = self.i_input + 1
   -- return the total (not averaged) cross entropy loss and the number of
   -- used in this mini-batch
   return loss_xe * num_samples, num_samples
end

-- Test (run forward only) on a single mini-batch.
function Mixer:test_one_batch(inputs, labels)
   local step = self.i_input % self.bptt
   self.pred[step + 1], self.hiddens[step + 1] =
      unpack(self.unrolled_nets[step + 1]:forward(
                {inputs, self.hiddens[step]}))
   local loss, nsamples =  self.nlls[step + 1]:forward(
      self.pred[step + 1][2], labels)
   self.i_input = self.i_input + 1
   return loss * nsamples, nsamples
end

function Mixer:set_generation_vars(nrstepsinit, mbsz, reward_func)
    local num_steps = self.bptt - self.nrstepsinit + 1
    self.reward = torch.Tensor(mbsz, num_steps):type(self.type)
    self.reward_func = reward_func
    self:set_nrstepsinit(nrstepsinit)
    self.pred = {}
    self.input2reward = {}
    self.indexes = torch.Tensor(mbsz):type(self.type)
    self.indexes_past = torch.Tensor(mbsz):type(self.type)
end

function Mixer:squeeze_but_keep_tensor(x)
    if x:nElement() == 1 then
        return x:dim() == 1 and x:type(self.type) or x:view(1):type(self.type)
    else
        return x:squeeze():type(self.type)
    end
end

-- Run the model forward like at training time but wihout backpropping.
-- This is used to evaluate the quality of generations.
function Mixer:eval_generation(input, label, maxgen)
   local loss_xe = 0 -- XENT
   local num_samples = 0 -- because of padding this maybe less than mbsz
   local num_samples_rf = nil
   local step = self.i_input % self.bptt
   self.inputs[step] = {}
   for k, v in pairs(input) do
       self.inputs[step][k] = v:clone()
   end
   if step >= self.nrstepsinit then
       self.inputs[step][1] = self:squeeze_but_keep_tensor(self.pred[step][1])
   end
   -- pred stores: sampled word, logprob scores, output of encoder
   -- fprop 1 step through RNN
   self.pred[step + 1], self.hiddens[step + 1] =
      unpack(self.unrolled_nets[step + 1]:forward(
                 {self.inputs[step], self.hiddens[step]}))
   if maxgen then -- replace sample with argmax
      local _, indx = self.pred[step + 1][2]:max(2)
      self.pred[step + 1][1]:copy(indx) -- overwrite sample with argmax
   end
   self.labels[step + 1] = label
   if step + 1 < self.nrstepsinit then
       self.pred[step + 1][1]:copy(label)
   else
       if step > 0 then
           self:overwrite_prediction(self.pred[step][1], self.pred[step + 1][1])
       end
   end
   self.pred_rf[step + 1][1] =
       self:squeeze_but_keep_tensor(self.pred[step + 1][1])
   -- and through cross entropy loss for next symbol
   if (step < self.nrstepsinit) then
      -- do not report xent loss after the initialization
      loss_xe, num_samples = self.nlls[step + 1]:forward(
         self.pred[step + 1][2], label)
   end
   local bleu = nil
   -- evaluate sentence.
   if step + 1 == self.bptt then
       -- work at the corpus level, collect counts
       self.criterion:get_counts_corpus(self.labels, self.pred_rf)
   end
   self.i_input = self.i_input + 1
   return loss_xe * num_samples, num_samples, bleu, num_samples_rf
end

function Mixer:weight_xent()
    return self.wxent
end

function Mixer:reset_reward()
    self.criterion:reset_reward()
end

function Mixer:get_corpus_score()
    return self.criterion:get_corpus_score()
end

function Mixer:sum()
    return self.param:sum()
end

function Mixer:training_mode()
    self.criterion:training_mode()
end

function Mixer:test_mode()
    self.criterion:test_mode()
end
