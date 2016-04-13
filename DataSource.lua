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

require 'torch'
require 'paths'
require 'math'
require 'xlua'
local tds = require('tds')

local DataSource  = torch.class('DataSource')

-- Data provider class that takes a binary tokenized dataset,
-- and provides mini-batches.
function DataSource:__init(config)
    self.batch_size = config.batch_size
    self.root = config.root_path
    self.dtype = config.data_type
    self.tdict = config.dct_target
    self.sdict = config.dct_source
    self.sepidx = self.tdict.separatorIndex
    self.bin_thresh = config.bin_thresh
    self.seqlength = config.sequence_length
    self.padidx_target = self.tdict.paddingIndex
    self.padidx_source = self.sdict.paddingIndex
    self.all_sources = torch.load(paths.concat(self.root,
                                               self.dtype .. '.sources.th7'))
    self.all_targets = torch.load(paths.concat(self.root,
                                               self.dtype .. '.targets.th7'))
    -- gather the shard ids
    self.shard_ids = {}
    local ctr = 0
    for i, v in pairs(self.all_targets) do
        local gross_size = v:size(1)
        if gross_size >= self.bin_thresh then
            ctr = ctr + 1
            self.shard_ids[ctr] = i
        end
    end
    -- create a permutation vector of the shards
    if self.dtype == 'train' then
        self.perm_vec = torch.randperm(#self.shard_ids)
    else
        self.perm_vec = torch.range(1, #self.shard_ids)
    end
    self.curr_shard_num = 0
    self.curr_shard_id = -1
    self.nshards = #self.shard_ids
    collectgarbage()
end

function DataSource:reset()
    self.curr_shard_num = 0
    self.curr_shard_id = -1
end

-- This function returns one "shard" corresponding to context
-- sentences of a single length. ie. a tensor of shard_length*batch_size
-- elements. Each slice alongside the second dimension represents
-- consecutive words and begins with a BoS (beginning of sentence).
-- They have fixed number of words and can be cut in the middle of a sentence
-- at the end.
function DataSource:get_next_shard()
    local id = self.curr_shard_num % self.nshards + 1
    local pid = self.perm_vec[id]
    local shard_id = self.shard_ids[pid]
    local twords = self.all_targets[shard_id]:size(1)
    self.curr_shard_num = self.curr_shard_num + 1
    -- keep looping over bins until you get one with more words in
    -- target set than the bin_thresh
    while twords < self.bin_thresh do
        id = self.curr_shard_num % self.nshards + 1
        pid = self.perm_vec[id]
        shard_id = self.shard_ids[pid]
        twords = self.all_targets[shard_id]:size(1)
        self.curr_shard_num = self.curr_shard_num + 1
    end
    return self:get_shard(pid)
end

-- Returns the number of shards in a set. See get_shard for more details.
function DataSource:get_nshards()
    return self.nshards
end


-- This function returns things associated with one shard
-- of the dataset, whose id is given by snum. In particular it returns:
--   1. a tensor (or table) corresponding to the current shard,
--      such that inputs[i] is the i-th training sample
--   2. a tensor corresponding to the current shard,
--      such that labels[i] is the i-th label
--   3. the size of the minibatch
--   4. the number of batches in the current shard
--   5. a tensor indicating whether the i-th sample is a beginning of a sequence
--     (+1) or whether it corresponds to the PAD token (-1) or actual work token
--     (0).
function DataSource:get_shard(snum)
    self.curr_shard_id = self.shard_ids[snum]
    self.curr_target = self.all_targets[self.curr_shard_id]
    self.curr_source = self.all_sources[self.curr_shard_id]
    -- first get all the sentences out
    local gross_size = self.curr_target:size(1)
    local sentenceidx = tds.hash()
    local num_sentences = 0
    local ww = 1
    while ww <= gross_size do
        if self.curr_target[ww][1] == self.sepidx then
            -- store the index of the separator token
            num_sentences = num_sentences + 1
            sentenceidx[num_sentences] = ww
        end
        ww = ww + 1
    end
    local info = torch.Tensor(2, num_sentences)
    for cc = 1, num_sentences - 1 do
        info[1][cc] = sentenceidx[cc] -- the start_idx
        info[2][cc] = sentenceidx[cc + 1] - sentenceidx[cc] -- length
    end
    -- compute start_idx and length of last sentence
    info[1][num_sentences] = sentenceidx[num_sentences]
    info[2][num_sentences] = gross_size - sentenceidx[num_sentences] + 1
    -- now construct minibatches by picking sentences (titles) at random
    -- and filling each sample in the minibatch in sequence
    self.num_batches = math.ceil(num_sentences / self.batch_size)
    -- input titles
    self.curr_target_shard =
        torch.Tensor(self.num_batches * self.seqlength,
                     self.batch_size):fill(self.padidx_target)
    -- labels
    self.curr_target_shard_lab =
        torch.Tensor(self.num_batches * self.seqlength,
                     self.batch_size):fill(self.padidx_target)
    -- matrix storing source (article) vector associated with title words
    self.curr_source_len = self.curr_shard_id
    self.curr_source_shard = self.curr_source_shard or torch.LongTensor()
    self.curr_source_shard:resize(self.num_batches * self.seqlength,
                                   self.batch_size, self.curr_source_len)
    self.curr_source_shard:fill(self.padidx_source)
    -- now load the various matrices with word ids
    local perm
    if self.dtype == 'train' then
        perm = torch.randperm(num_sentences)
    else
        perm = torch.range(1, num_sentences)
    end
    for ss = 1, num_sentences do
        local curr_start = info[1][perm[ss]]
        local curr_length = math.min(info[2][perm[ss]], self.seqlength)
        if curr_length > 0 then
            local row =
                math.floor((ss - 1) / self.batch_size) * self.seqlength + 1
            local col = (ss - 1) % self.batch_size + 1
            local target_ids =
                self.curr_target:narrow(1, curr_start, curr_length)
            -- load the target matrix
            self.curr_target_shard:select(2, col):narrow(
                1, row, curr_length):copy(target_ids:select(2, 1))
            -- load the source matrix
            local source_id = target_ids[1][2] -- source index
            local source = self.curr_source[source_id] -- source words
            for i = 1, curr_length do
                self.curr_source_shard[{row + i - 1, col, {}}]:copy(source)
            end

            -- load the label matrix
            curr_length = math.min(info[2][perm[ss]], self.seqlength + 1)
            if curr_length > 1 then
                self.curr_target_shard_lab:select(2, col):narrow(
                    1, row, curr_length - 1):copy(
                self.curr_target:select(2, 1):narrow(
                    1, curr_start + 1, curr_length - 1))
                -- add eos as the last word in label if sentence
                -- is shorter than sequence length
                if curr_length < self.seqlength then
                    self.curr_target_shard_lab:select(2, col)[
                        row + curr_length - 1] = self.sepidx
                end
            end
        else
            error('found empty sentence!')
        end
    end
    collectgarbage()
    return self:get_shard_info_clm(self.curr_target_shard,
                                   self.curr_target_shard_lab,
                                   self.curr_source_shard)
end

function DataSource:get_shard_info_clm(cstgt, cslab, cssrc)
    local inputs = {}
    inputs.target = cstgt
    inputs.source = cssrc
    inputs.cposition = torch.LongTensor():resizeAs(inputs.source)
    local winsz = inputs.cposition:size(3)
    for i = 1, winsz do
        inputs.cposition[{{}, {}, i}]:fill(i)
    end

    -- create a metatable associated with the inputs
    inputs.mt = {}
    inputs.mt.__index = function(self, index)
        return {self.target[index], self.source[index], self.cposition[index]}
    end
    setmetatable(inputs, inputs.mt)

    function inputs:size()
        return self.target:size(1)
    end

    function inputs:type(tp)
        self.target = self.target:type(tp)
        self.source = self.source:type(tp)
        self.cposition = self.cposition:type(tp)
        return self
    end

    local labels = cslab
    local batch_size = cstgt:size(2)
    local nbatches = inputs:size(1)

    return inputs, labels, batch_size, nbatches
end
