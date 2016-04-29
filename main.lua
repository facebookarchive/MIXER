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
-- Script that launches training and testing of a summarization model
-- using MIXER as described in:
-- http://arxiv.org/abs/1511.06732 (ICLR 2016)
require('xlua')
require('nn')
require('nngraph')
require('cutorch')
require('cunn')
paths.dofile('Trainer.lua')
paths.dofile('Mixer.lua')
paths.dofile('ReinforceCriterion.lua')
paths.dofile('DataSource.lua')
local mdls = paths.dofile('model_factory.lua')
paths.dofile('reward_factory.lua')

torch.manualSeed(1111)
cutorch.manualSeed(1111)
local cmd = torch.CmdLine()
cmd:option('-datadir', 'data', 'path to binarized training data')
cmd:option('-lr', 0.2, 'learning rate')
cmd:option('-gparamclip', 10, 'clipping threshold of parameter gradients')
cmd:option('-bsz', 32, 'batch size')
cmd:option('-nhid', 256, 'number of hidden units')
cmd:option('-bptt', 25, 'number of backprop steps through time')
cmd:option('-deltasteps', 3,
           'increment of number of words we predict using REINFORCE at next' ..
               ' round')
cmd:option('-nepochs', 5,
           'number of epochs of each stage of REINFORCE training')
cmd:option('-epoch_xent', 25,
           'number of epochs we do with pure XENT to initialize the model')
cmd:option('-devid', 1, 'GPU device id')
cmd:option('-reward', 'bleu', 'reward type: bleu|rouge')
local config = cmd:parse(arg)
cutorch.setDevice(config.devid)
-- building configuration hyper-parameters for Trainer, Model and Dataset.
config.trainer = {
    bptt = config.bptt,
    n_epochs = config.nepochs,
    initial_learning_rate = config.lr,
    save_dir = './backups/'} -- directory where we save checkpoints
config.model = {
    n_hidden = config.nhid,
    batch_size = config.bsz,
    bptt = config.bptt,
    grad_param_clip = config.gparamclip,
    reward = config.reward}

-- load data
local path2data = config.datadir
-- download the data if its not already in the data directory.
if not (paths.dirp(path2data) and
            paths.filep(paths.concat(path2data, 'dict.target.th7')) and
            paths.filep(paths.concat(path2data, 'dict.source.th7'))) then
    print('[[ Data not found: fetching a fresh copy and running tokenizer]]')
    os.execute('./prepareData.sh')
end
-- load target and source dictionaries and add padding token
local dict_target = torch.load(paths.concat(path2data, 'dict.target.th7'))
local dict_source = torch.load(paths.concat(path2data, 'dict.source.th7'))
-- add the padding index if using the aligned data source
dict_target.nwords = dict_target.nwords + 1
local padidx_target = dict_target.nwords
dict_target.index_to_symbol[padidx_target] = '<PAD>'
dict_target.symbol_to_index['<PAD>'] = padidx_target
dict_target.paddingIndex = padidx_target
dict_source.nwords = dict_source.nwords + 1
local padidx_source = dict_source.nwords
dict_source.index_to_symbol[padidx_source] = '<PAD>'
dict_source.symbol_to_index['<PAD>'] = padidx_source
dict_source.paddingIndex = padidx_source
local train_data = DataSource(
    {root_path  = path2data,
     data_type  = 'train',
     batch_size = config.bsz,
     bin_thresh = 800,
     sequence_length = config.bptt,
     dct_target = dict_target,
     dct_source = dict_source})
local valid_data = DataSource(
    {root_path  = path2data,
     data_type  = 'valid',
     batch_size = config.bsz,
     bin_thresh = 800,
     max_shard_len = 0,
     sequence_length = config.bptt,
     dct_target = dict_target,
     dct_source = dict_source})
local test_data = DataSource(
    {root_path  = path2data,
     data_type  = 'test',
     batch_size = config.bsz,
     bin_thresh = 800,
     max_shard_len = 0,
     sequence_length = config.bptt,
     dct_target = dict_target,
     dct_source = dict_source})

-- create and initialize the core net at a given time step
config.model.eosIndex = dict_target.separatorIndex
config.model.n_tokens = dict_target.nwords
config.model.paddingIndex = dict_target.paddingIndex
local unk_id = dict_target.symbol_to_index['<unk>']

local net, size_hid_layers = mdls.makeNetSingleStep(
    config.model, dict_target, dict_source)
config.model.size_hid_layers = size_hid_layers

-- create the criterion for the whole sequence
local compute_reward =
    RewardFactory(config.model.reward, config.bptt, config.model.n_tokens,
                  config.model.eosIndex, config.model.paddingIndex, unk_id,
                  config.bsz)
compute_reward:training_mode()
compute_reward:cuda()
local reinforce = nn.ReinforceCriterion(compute_reward, config.bptt,
                                        config.model.eosIndex,
                                        config.model.paddingIndex)
-- create and initialize the RNNreinforce model (replicating "net" over time)
local model = Mixer(config.model, net, reinforce)
local trainer = Trainer(config.trainer, model)
trainer:cuda()

print('Start training')
-- start by training using XENT only.
model:set_xent_weight(1)
local start_nrstepsinit = config.model.bptt
for nrstepsinit = start_nrstepsinit, 1, -config.deltasteps do
   print('nrstepsinit', nrstepsinit)
   trainer.save_dir = config.trainer.save_dir .. nrstepsinit .. '/'
   trainer:set_nrstepsinit(nrstepsinit)
   if nrstepsinit < config.model.bptt then
       model:set_xent_weight(0.5)
   end
   local num_epochs = config.trainer.n_epochs
   if (nrstepsinit == config.model.bptt) then
       num_epochs = config.epoch_xent
   end
   if nrstepsinit == 1 or nrstepsinit - config.deltasteps < 1 then
       num_epochs = 100 -- run forever at the very last iteration
   end
   trainer:run(num_epochs, train_data, valid_data, test_data)
    -- compute BLEU on validation set
   trainer:set_nrstepsinit(1)
   trainer:eval_generation(valid_data)
end
print('End of training.')
print('****************')
print('Evaluating now generation on the test set')
trainer:eval_generation(test_data)
