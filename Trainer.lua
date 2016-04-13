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
-- Trainer which calls FPROP/BPROP for each mini-batch, trains for the
-- specified number of epochs and runs the evaluation.
require 'math'
require 'sys'
require 'os'
require 'torch'
require 'xlua'
local utils = paths.dofile('util.lua')

local Trainer = torch.class('Trainer')

function Trainer:__init(config, model)
   self.model = model
   -- optimization hyper-parameters
   self.learning_rate = config.initial_learning_rate
   self.learning_rate_shrink = 1.2
   self.shrink_multiplier = 0.9999
   self.type = config.type
   self.save = config.save
   self.save_dir = config.save_dir
   self.verbose = config.verbose
end

function Trainer:cuda()
    self.type = 'torch.CudaTensor'
end

-- Train for one epochs over the whole dataset.
function Trainer:train(dset)
    dset:reset()
    self.model:reset_rf_vars()
    self.model:training_mode()
    local nshards = dset:get_nshards()
    local loss_per_shard = torch.zeros(nshards)
    local nsamples_per_shard = torch.zeros(nshards)
    local coeff_rf, output_rf, reward_predictor_error
    local num_samples_rf, tot_cumreward_err, tot_reward
    for j = 1, nshards do
        self.model:reset()
        local inputs, labels, batch_size, nbatches =
            dset:get_next_shard()
        inputs = inputs:type(self.type)
        labels = labels:type(self.type)
        for i = 1, nbatches do
            if i % 1000 == 0 then
                if sys.isNaN(self.model:sum()) then
                    print('Not a Number detected')
                    os.exit(0)
                end
            end
            -- this returns only the total XENT loss
            local loss_batch, nsamples_batch =
               self.model:train_one_batch(inputs[i], labels[i],
                                          self.learning_rate)
            loss_per_shard[j] = loss_per_shard[j] + loss_batch
            nsamples_per_shard[j] = nsamples_per_shard[j] + nsamples_batch
        end
        collectgarbage()
    end
    local entropy = (loss_per_shard:sum() / nsamples_per_shard:sum()) /
        (self.model:weight_xent() * math.log(2))
    num_samples_rf, tot_cumreward_err, tot_reward =
        self.model:get_rf_vars()
    coeff_rf = (num_samples_rf * self.model.criterion.weight)
    output_rf = (coeff_rf == 0) and 0 or tot_reward / coeff_rf
    reward_predictor_error = (num_samples_rf == 0)
        and 0 or tot_cumreward_err / num_samples_rf

    return entropy, output_rf, reward_predictor_error,
           nsamples_per_shard:sum(), num_samples_rf
end

-- Evaluate.
function Trainer:eval(dset)
    dset:reset()
    self.model:reset_rf_vars()
    local n_shards = dset:get_nshards()
    local loss_per_shard = torch.zeros(n_shards)
    local nsamples_per_shard = torch.zeros(n_shards)
    for sid = 1, n_shards do
        self.model:reset()
        local inputs, labels, batch_size, num_batches =
            dset:get_next_shard()
        inputs = inputs:type(self.type)
        labels = labels:type(self.type)
        for i = 1, num_batches do
            local loss_batch, nsamples_batch =
                self.model:test_one_batch(inputs[i], labels[i])
            loss_per_shard[sid] = loss_per_shard[sid] + loss_batch
            nsamples_per_shard[sid] = nsamples_per_shard[sid] + nsamples_batch
        end
        collectgarbage()
    end
    local entropy = (loss_per_shard:sum() / nsamples_per_shard:sum()) /
        (self.model:weight_xent() * math.log(2))
    return entropy, nsamples_per_shard:sum()
end


-- Runs training and testing for the specified number of epochs.
function Trainer:run(n_epoches, dset_train, dset_valid, dset_test)
   local last_val_err = 1e30
   local train_err = torch.zeros(n_epoches)
   local val_err = torch.zeros(n_epoches)
   local test_err = torch.zeros(n_epoches)
   local lr = torch.zeros(n_epoches)
   local time = torch.zeros(n_epoches)
   local n_words_xe, n_words_rf, output_rf, reward_pred_err
   -- save the untrained model
   if self.save and self.save_dir ~= nil then
       if paths.dirp(self.save_dir) == false then
           os.execute('mkdir -p ' .. self.save_dir)
       end
       print('*** saving the model ***')
       self.model:save(paths.concat(self.save_dir, 'model_0'))
   end

   for i = 1, n_epoches do
      local timer = torch.tic()
      lr[i] = self.learning_rate
      train_err[i], output_rf, reward_pred_err, n_words_xe, n_words_rf =
          self:train(dset_train)
      time[i] = torch.toc(timer)
      local ss1, ss2
      ss1 = string.format('\nEpoch: %d. Training time: %.2fs. ' ..
                              'WordsXE/s: %.2f, WordsRF/s: %.2f',
                          i, time[i],
                          n_words_xe / time[i], n_words_rf / time[i])
      ss2 = string.format(
          '\nTraining: Ent: %.5f || Ppl: %0.5f || ' ..
              'Avg. reward/token: %.3f || Cum. reward error: %.3f || ' ..
              'Nsamples Xent: %d || Nsamples Rf: %d',
          train_err[i], math.pow(2, train_err[i]),
          output_rf, reward_pred_err,
          n_words_xe, n_words_rf)
      io.write(ss1)
      io.write(ss2)
      io.flush()

      -- save the trained model
      if paths.dirp(self.save_dir) == false then
          os.execute('mkdir -p ' .. self.save_dir)
      end
      self.model:save(paths.concat(self.save_dir, 'model_' .. i))

      -- evaluate model on the validation set
      val_err[i] = self:eval(dset_valid)
      io.write(string.format(
                   '\nValidation: Ent: %.5f || Ppl: %0.5f',
                   val_err[i], math.pow(2, val_err[i])))
      io.flush()

      -- decrease learning rate if needed
      if (val_err[i] > last_val_err * self.shrink_multiplier) then
          -- anneal learning rate when valid error does not decrease enough
          self.learning_rate =
              self.learning_rate / self.learning_rate_shrink
          io.write('\nDecreasing the learning rate to '
                       .. self.learning_rate)
      else
          last_val_err = val_err[i]
      end

      -- save the logs of accuracy and time
      torch.save(paths.concat(self.save_dir, 'model.log'),
                 {train_err = train_err, test_err = test_err,
                  valid_err = val_err, lr = lr, output_rf = output_rf,
                  reward_pred_err = reward_pred_err, time = time, epoch = i})

      if self.learning_rate < 1e-4 then
          io.write('\nExiting because the learning rate is too small:' ..
                       self.learning_rate .. '\n')
          break
      end
   end
   return train_err[n_epoches], val_err[n_epoches], test_err[n_epoches],
          output_rf, reward_pred_err
end

-- Run evaluation on both validation and test set.
function Trainer:run_evaluate()
    -- evaluate model on the validation set
    local val_err = self:eval('valid')
    io.write(string.format('\nValidation: Ent: %.5f || Ppl: %0.5f',
                           val_err, math.pow(2, val_err)))
    io.flush()
    -- evaluate model on the test set
    local test_err = self:eval('test')
    io.write(string.format('\nTesting: Ent: %.5f || Ppl: %0.5f',
                           test_err, math.pow(2, test_err)))
    io.flush()
end

-- Evaluate the generation.
function Trainer:eval_generation(dset, maxgen)
    local use_max = (maxgen == nil) and true or maxgen
    dset:reset()
    self.model:reset_rf_vars()
    local n_shards = dset:get_nshards()
    local loss_per_shard = torch.zeros(n_shards)
    local nsamples_per_shard = torch.zeros(n_shards)
    local tot_rewards = 0
    local num_samples_rf = 0
    self.model:reset_reward()
    self.model:test_mode()
    for sid = 1, n_shards do
        self.model:reset()
        if self.verbose == 2 then
            print('-- total reward: ' .. tot_rewards/num_samples_rf)
        end
        local inputs, labels, batch_size, nbatches =
            dset:get_next_shard()
        inputs = inputs:type(self.type)
        labels = labels:type(self.type)
        for i = 1, nbatches do
            local loss_batch, nsamples_batch, loss_rf, nsamples =
               self.model:eval_generation(
                    inputs[i], labels[i], use_max)
            loss_per_shard[sid] = loss_per_shard[sid] + loss_batch
            nsamples_per_shard[sid] = nsamples_per_shard[sid] + nsamples_batch
        end
        collectgarbage()
    end
    -- compute bleu across the whole corpus
    local bleu
    bleu, num_samples_rf = self.model:get_corpus_score()
    tot_rewards = bleu * num_samples_rf
    local entropy = (loss_per_shard:sum() / nsamples_per_shard:sum()) /
        (self.model:weight_xent() * math.log(2))
    local perplexity = math.pow(2, entropy)
    local ns_xe = nsamples_per_shard:sum()
    io.write(string.format('\nEvaluating generation using %s', use_max and
                               'ARGMAX' or 'SAMPLING'))
    io.write(string.format(
                 '\nXENT Entropy: %.5f || Perplexity: %.5f; # words: %d',
                 entropy, perplexity, ns_xe))
    io.write(string.format(
                 '\n Avg. reward at step %d (after initializing for %d steps)'
                     .. ' is %.5f; # words %d', self.model.bptt,
                 self.model.nrstepsinit, tot_rewards / num_samples_rf,
                 num_samples_rf))
    print('')
    return tot_rewards / num_samples_rf, perplexity
end

function Trainer:set_nrstepsinit(val)
    self.model:set_nrstepsinit(val)
end
