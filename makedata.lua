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
-- Script to create torch data set from text files of source-target pair
-- sentences. It creates the source and target dictionaries, and the train,
-- valid and test torch files.
require 'math'
require 'sys'
require 'os'
require 'torch'
require 'xlua'

local tok = paths.dofile('tokenizer.lua')

torch.manualSeed(1)

cmd = torch.CmdLine()
cmd.argseparator = '_'
cmd:text()
cmd:text('Make dictionary and datasets')
cmd:text()
cmd:text('Options:')
cmd:option('-srcDir', 'prep', 'path to pre-processed data.')
cmd:option('-dstDir', 'data', 'path to where dictionaries and datasets ' ..
  'should be written.')
cmd:option('-shuff', true, 'shuffle sentences in training data or not')
cmd:option('-threshold', 3, 'remove words appearing less than threshold')
cmd:option('-isvalid', true, 'generate the validation set')
cmd:option('-istest', true, 'generate the test set')
cmd:text()
local opt = cmd:parse(arg)

if not paths.dirp(opt.dstDir) then os.execute('mkdir -p ' .. opt.dstDir) end

local config_data = {
    root_path = opt.srcDir,
    dest_path = opt.dstDir,
    threshold = opt.threshold,
    targets = {train = 'train.de-en.en',
               valid = 'valid.de-en.en',
               test  = 'test.de-en.en'},
    sources = {train = 'train.de-en.de',
               valid = 'valid.de-en.de',
               test  = 'test.de-en.de'},
}

-- build and save the dictionaries
local tdict_path = paths.concat(opt.dstDir, 'dict.target.th7')
local sdict_path = paths.concat(opt.dstDir, 'dict.source.th7')
local target_dict, source_dict

print('-- building target dictionary')
local train_target = paths.concat(opt.srcDir, config_data.targets['train'])
target_dict = tok.build_dictionary(train_target, config_data.threshold)
torch.save(tdict_path, target_dict)

print('-- building source dictionary')
local train_source = paths.concat(opt.srcDir, config_data.sources['train'])
source_dict = tok.build_dictionary(train_source, config_data.threshold)
torch.save(sdict_path, source_dict)

-- now create the binned training data: target sentences corresponding to
-- each length of the source sentence are binned together
local train_targets_path = paths.concat(config_data.dest_path,
                                        'train.targets.th7')
local train_sources_path = paths.concat(config_data.dest_path,
                                        'train.sources.th7')
local valid_targets_path = paths.concat(config_data.dest_path,
                                        'valid.targets.th7')
local valid_sources_path = paths.concat(config_data.dest_path,
                                        'valid.sources.th7')
local test_targets_path = paths.concat(config_data.dest_path,
                                        'test.targets.th7')
local test_sources_path = paths.concat(config_data.dest_path,
                                        'test.sources.th7')
print('tokenizing train...')
local train_targets, train_sources = tok.tokenize(config_data, 'train',
                                                  target_dict, source_dict,
                                                  opt.shuff)
print('tokenizing valid...')
local valid_targets, valid_sources = tok.tokenize(config_data, 'valid',
                                                  target_dict, source_dict,
                                                  false)
print('tokenizing test...')
local test_targets, test_sources   = tok.tokenize(config_data, 'test',
                                                  target_dict, source_dict,
                                                  false)
torch.save(train_targets_path, train_targets)
torch.save(train_sources_path, train_sources)
torch.save(valid_targets_path, valid_targets)
torch.save(valid_sources_path, valid_sources)
torch.save(test_targets_path,  test_targets)
torch.save(test_sources_path,  test_sources)
