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
require 'math'
require('xlua')
local utils = paths.dofile('util.lua')
local threads = require('threads')
local RewardFactory = torch.class("RewardFactory")

-- This class returns an object for computing the reward at a
-- given time step.
-- reward_type  type of reward, either ROUGE or BLEU
-- start  index of time step at which we start computing the reward
-- bptt   the maximum length of a sequence.
-- dict_size  size of the dictionary
-- eos_indx is the id of the end of sentence token. Symbols after
--  the first occurrence of eos (if any) are skipped.
-- pad_indx is the id of the padding token
-- mbsz mini-batch size
function RewardFactory:__init(reward_type, bptt, dict_size,
                              eos_indx, pad_indx, unk_indx, mbsz)
   self.reward_type = reward_type
   self.start = 1
   self.dict_size = dict_size
   self.eos_indx = eos_indx
   self.pad_indx = pad_indx
   if unk_indx == nil then
       print('dictionary does not have <unk>, ' ..
             'we are not skipping then while computing BLEU')
       self.unk_indx = -1
   else
       self.unk_indx = unk_indx
   end
   self.mbsz = mbsz
   self.reward_val = torch.Tensor(mbsz)
   -- auxiliary vars
   self.input_pads = torch.Tensor(mbsz)
   self.target_pads = torch.Tensor(mbsz)
   self.inputt = torch.Tensor(bptt - self.start + 1)
   self.targett = torch.Tensor(bptt - self.start + 1)
   self.reset = torch.Tensor(mbsz)
   self.target = torch.Tensor(bptt, mbsz)
   self.input = torch.Tensor(bptt, mbsz)
   -- Since counting works on cpu, we speed up by multi-threading.
   self.nthreads = 8
   threads.serialization('threads.sharedserialize')
   self.pool = threads.Threads(self.nthreads)
   self.pool:specific(true)
   for i = 1, self.nthreads do
       self.pool:addjob(
           i,
           function()
               require 'xlua'
               local utils = paths.dofile('util.lua')
               require('cutorch')
               require('math')
            end
         )
   end
   self.pool:specific(false)
   self.order = 4 -- we compute up to 4-grams
   self.score = torch.zeros(self.order, 3)
   self.sentence_bleu = 0
   self.length_input = 0
   self.length_target = 0
   self.counter = 0
   self.smoothing_val = 1
   self.adjust_bp = true
end

function RewardFactory:test_mode()
    self.smoothing_val = 0
    self.adjust_bp = false
end

-- BLEU: Smooth score and adjust brevity penalty
-- only at training time since we work at the sentence
-- level.
function RewardFactory:training_mode()
    self.smoothing_val = 1
    self.adjust_bp = true
end

function RewardFactory:reset_vars()
    self.length_input = 0
    self.length_target = 0
    self.counter = 0
    self.score:fill(0)
    self.sentence_bleu = 0
end

function RewardFactory:set_start(val)
    assert(val > 0)
    self.start = val
end

function RewardFactory:type(tt)
   self.reset = self.reset:type(tt)
end

function RewardFactory:cuda()
   self.reset = self.reset:cuda()
end

-- target is a table. Each entry is a tensor of size mini-batch size
-- storing the reference at a certain time step.
-- input is a table of tables. Each table stores in its
-- first entry the word we have sampled. The second entry stores
-- an estimate of cumulative reward, and it is not used here.
-- tt is the time step at which we wish to compute the reward.
function RewardFactory:get_reward(target, input, tt)
   self.reward_val:fill(0)
   if self.reward_type == 'rouge' then
       -- Rouge @ N where N is equal to self.order computes the score as:
       -- number of matching ngrams / number of ngrams in the reference.
       function compute_rouge(target, input, args, i, tt)
          local bptt = target:size(1)
          -- get local copy of class member variables
          local start = args.start
          local dict_size = args.dict_size
          local eos_indx = args.eos_indx
          local pad_indx = args.pad_indx
          local unk_indx = args.unk_indx
          local mbsz = args.mbsz
          local reward_val = args.reward_val
          local inputt = torch.Tensor(bptt - start + 1)
          local targett = torch.Tensor(bptt - start + 1)
          local nthreads = args.nthreads
          local order = args.order
          local num_samples = math.floor(mbsz / nthreads)
          local first = (i - 1) * num_samples + 1
          local last = (i < nthreads) and first + num_samples - 1 or mbsz
          for ss = first, last do
              -- compute the length of the input and target sequences
              -- default values if eos is not found is bptt
              local target_length = bptt
              local input_length = bptt
              for step = 1, bptt do
                  if target[step][ss] == eos_indx then
                      target_length = step - 1
                      break
                  end
              end
              for step = 1, bptt do
                  if input[step][ss] == eos_indx then
                      input_length = step - 1
                      break
                  end
              end
              -- some samples in the minibatch may just have
              -- PAD token everywhere.
              if input[1][ss] == pad_indx then
                  input_length = 0
              end
              if target[1][ss] == pad_indx then
                  target_length = 0
              end
              assert(target_length >= 0 and input_length >= 0)
              local min_len = math.min(input_length - start + 1,
                                       target_length - start + 1)
              -- non-zero reward only if both target and generates strings
              -- are longer than self.order and the generated sentece is done
              if (tt == nil or tt == math.min(input_length + 1, bptt))
                  and min_len >= order
              then
                  local score = 0
                  local offset = math.min(order - 1, start - 1)
                  local eff_seq_length_input = input_length - start + 1 +
                      offset
                  inputt:resize(eff_seq_length_input)
                  local eff_seq_length_target = target_length - start + 1 +
                      offset
                  targett:resize(eff_seq_length_target)
                  inputt:copy(
                      input:select(2, ss):narrow(
                          1, start - offset, eff_seq_length_input))
                  targett:copy(
                      target:select(2, ss):narrow(
                          1, start - offset, eff_seq_length_target))
                  local counts_input = {}
                  local counts_target = {}
                  local curr_offs = math.max(offset + 1 - order + 1, 1)
                  counts_input = utils.get_counts(
                      inputt:narrow(1, curr_offs,
                                    eff_seq_length_input - curr_offs + 1),
                      order, dict_size)
                      counts_target = utils.get_counts(
                          targett:narrow(1, curr_offs,
                                         eff_seq_length_target - curr_offs + 1),
                          order, dict_size, unk_indx)
                      score =
                          utils.compute_recall(counts_input, counts_target)
                  reward_val[ss] = score
              end -- reward > 0 only at the very end of the sequence only
          end -- end loop over samples
          collectgarbage()
      end
      for cc = 1, #target do
          self.target:select(1, cc):copy(target[cc])
          self.input:select(1, cc):copy(input[cc][1])
      end
      local args = {start = self.start, dict_size = self.dict_size,
              eos_indx = self.eos_indx,
              pad_indx = self.pad_indx, unk_indx = self.unk_indx,
              mbsz = self.mbsz, reward_val = self.reward_val,
              nthreads = self.nthreads, order = self.order}
      for i = 1, self.nthreads do
          self.pool:addjob(compute_rouge, function () end,
                           self.target, self.input, args, i, tt)
      end
      self.pool:synchronize()
      return self.reward_val
   elseif self.reward_type == 'bleu' then
      -- DISCLAIMER: the score is smoothed
      -- because our sequences are short and it's likely that some scores are 0
      -- (which would make the geometric mean be 0 as well). Smoothing should be
      -- used only at training time (since at test time we evaluate at the
      -- corpus level).
      -- NOTE: target and input are tables with the same number of entries,
      -- however, each sequence can have an eos at different time steps
      -- (so effectively we do not assume that input and target have the same
      -- length).
      function compute_bleu(target, input, tt, args, i)
          local bptt = target:size(1)
          -- get local copy of class member variables
          local start = args.start
          local dict_size = args.dict_size
          local eos_indx = args.eos_indx
          local pad_indx = args.pad_indx
          local unk_indx = args.unk_indx
          local mbsz = args.mbsz
          local reward_val = args.reward_val
          local inputt = torch.Tensor(bptt - start + 1) -- args.inputt
          local targett = torch.Tensor(bptt - start + 1) -- args.targett
          local nthreads = args.nthreads
          local order = args.order
          local smoothing_val = args.smoothing_val
          local adjust_bp = args.adjust_bp
          local num_samples = math.floor(mbsz / nthreads)
          local first = (i - 1) * num_samples + 1
          local last = (i < nthreads) and first + num_samples - 1 or mbsz

          for ss = first, last do
              -- compute the length of the input and target sequences
              -- default values if eos is not found is bptt
              local target_length = bptt
              local input_length = bptt
              for step = 1, bptt do
                  if target[step][ss] == eos_indx then
                      target_length = step - 1
                      break
                  end
              end
              for step = 1, bptt do
                  if input[step][ss] == eos_indx then
                      input_length = step - 1
                      break
                  end
              end
              -- some samples in the minibatch may just have
              -- PAD token everywhere.
              if input[1][ss] == pad_indx then
                  input_length = 0
              end
              if target[1][ss] == pad_indx then
                  target_length = 0
              end
              assert(target_length >= 0 and input_length >= 0)
              -- we go up to 4-grams.
              -- Note: if eos is detected before self.start, then
              -- reward is 0.
              local n = math.min(order, input_length - start + 1,
                                 target_length - start + 1)
              -- non-zero reward only if an eos has been found in the input
              -- or we reached the max length. We add 1 because input_length
              -- is the length up to the symbol before eos but we want to give
              -- reward when we encounter eos.
              if tt == math.min(input_length + 1, bptt) and n > 0 then
                  local score = torch.Tensor(n):fill(0)
                  -- extracts the ending part of the input and target sequences,
                  -- taking into account ngrams that overlap between the
                  -- conditioning part and the generated part.
                  local eff_seq_length_input = input_length - start + 1 +
                      -- consider ngrams overlapping with part we condition upon
                      -- but be careful not to run out of words.
                      math.min(n - 1, start - 1)
                  inputt:resize(eff_seq_length_input)
                  local eff_seq_length_target = target_length - start + 1 +
                      math.min(n - 1, start - 1)
                  targett:resize(eff_seq_length_target)
                  -- copy data from tables to tensors
                  for step = 1, eff_seq_length_input do
                      inputt[step] = input[
                          start + step - 1 - math.min(n - 1, start - 1)][
                          ss]
                  end
                  for step = 1, eff_seq_length_target do
                      targett[step] = target[
                          start + step - 1 - math.min(n - 1, start - 1)][ss]
                  end
                  local counts_input = {} -- stores counts hashes for each n
                  local counts_target = {}
                  local offset = math.min(n - 1, start - 1)
                  for nn = 1, n do
                      -- restrict counting to ngrams that depend on the
                      -- generated sequence (yet potentially overlapping with
                      -- the conditioning part of the sequence).
                      local curr_offs = math.max(offset + 1 - nn + 1, 1)
                      counts_input[nn] = utils.get_counts(
                          inputt:narrow(1, curr_offs,
                                        eff_seq_length_input - curr_offs + 1),
                          nn, dict_size)
                      counts_target[nn] = utils.get_counts(
                          targett:narrow(1, curr_offs,
                                         eff_seq_length_target - curr_offs + 1),
                          nn, dict_size, unk_indx)
                      score[nn] =
                          utils.compute_score(
                              counts_input[nn], counts_target[nn],
                              smoothing_val)
                  end
                  -- compute bleu score: exp(1/N sum_n log score_n)
                  reward_val[ss] = score:log():sum(1):div(n):exp()
                  -- add brevity penalty
                  local bp = 1
                  if input_length < target_length then
                      bp =
                          math.exp(1 - (target_length +
                                            (adjust_bp and smoothing_val or 0))
                                       / input_length)
                  end
                  reward_val[ss] = reward_val[ss] * bp
              end -- reward > 0 only at the very end of the sequence only
          end -- end loop over samples
          collectgarbage()

      end
      for cc = 1, #target do
          self.target:select(1, cc):copy(target[cc])
          self.input:select(1, cc):copy(input[cc][1])
      end
      local args = {start = self.start, dict_size = self.dict_size,
              eos_indx = self.eos_indx,
              pad_indx = self.pad_indx, unk_indx = self.unk_indx,
              mbsz = self.mbsz, reward_val = self.reward_val,
              inputt = self.inputt, targett = self.targett,
              nthreads = self.nthreads, order = self.order,
              smoothing_val = self.smoothing_val, adjust_bp = self.adjust_bp}
      for i = 1, self.nthreads do
          self.pool:addjob(compute_bleu, function () end,
                           self.target, self.input, tt, args, i)
      end
      self.pool:synchronize()
      return self.reward_val
   else
       error('not implemented yet')
   end

end

function RewardFactory:num_samples(target, input)
   if self.reward_type == 'bleu' or self.reward_type == 'rouge' then
      -- return the number of sentences since BLEU is a sentence level score.
      -- we count the number of sentence by removing those that have PAD at
      -- "start", all others are valid.
      self.reset:ne(target[self.start], self.pad_indx)
      return self.reset:sum()
   else
      error(self.reward_type .. ' has not been implemented')
   end
end

function RewardFactory:get_counts_corpus(target, input)
    -- count ngrams of all sentences in the corpus.
    if self.reward_type == 'bleu' then
       for cc = 1, #target do
           self.target:select(1, cc):copy(target[cc])
           self.input:select(1, cc):copy(input[cc][1])
       end
       local bptt = self.target:size(1)
       for ss = 1, self.mbsz do
           -- compute the length of the input and target sequences
           -- default values if eos is not found is bptt
           local target_length = bptt
           local input_length = bptt
           for step = 1, bptt do
               if self.target[step][ss] == self.eos_indx then
                   target_length = step - 1
                   break
               end
           end
           for step = 1, bptt do
               if self.input[step][ss] == self.eos_indx then
                   input_length = step - 1
                   break
               end
           end
           -- some samples in the minibatch may just have
           -- PAD token everywhere.
           if self.input[1][ss] == self.pad_indx then
               input_length = 0
           end
           if self.target[1][ss] == self.pad_indx then
               target_length = 0
           end
           assert(target_length >= 0 and input_length >= 0)
           local n = math.min(self.order, input_length - self.start + 1,
                              target_length - self.start + 1)
           -- non-zero reward only if an eos has been found in the input
           -- or we reached the max length. We add 1 because input_length
           -- is the length up to the symbol before eos but we want to give
           -- reward when we encounter eos.
           if n > 0 then
               self.counter = self.counter + 1
               -- number of tokens taken from part of the sequence we condition
               -- upon. When start > n, we take extra n-1 tokens to account for
               -- n-grams overlapping with the part of the sequence we condition
               -- upon
               local offset = math.min(n - 1, self.start - 1)
               local eff_seq_length_input = input_length - self.start + 1 +
                   offset
               self.inputt:resize(eff_seq_length_input)
               local eff_seq_length_target = target_length - self.start + 1 +
                   offset
               self.targett:resize(eff_seq_length_target)
               self.length_input = self.length_input + input_length
                   - self.start + 1
               self.length_target = self.length_target + target_length
                   - self.start + 1
               -- copy data from tables to tensors
               self.inputt:copy(
                   self.input:select(2, ss):narrow(
                       1, self.start - offset, eff_seq_length_input))
               self.targett:copy(
                   self.target:select(2, ss):narrow(
                       1, self.start - offset, eff_seq_length_target))
               local counts_input = {} -- stores counts hashes for each n
               local counts_target = {}
               for nn = 1, n do
                   -- if start > n, then this is n - nn + 1.
                   -- otherwise, we take into account the left border effect.
                   local curr_offs = math.max(offset + 1 - nn + 1, 1)
                   counts_input[nn] = utils.get_counts(
                       self.inputt:narrow(1, curr_offs,
                                          eff_seq_length_input - curr_offs + 1),
                       nn, self.dict_size, nil)
                   counts_target[nn] = utils.get_counts(
                       self.targett:narrow(
                           1, curr_offs, eff_seq_length_target - curr_offs + 1),
                       nn, self.dict_size, self.unk_indx) -- skip UNKs
                   utils.compute_precision(
                       self.score[nn], counts_input[nn],
                       counts_target[nn], self.smoothing_val)
               end
               local curr_bleu =
                   self.score:select(2, 3):narrow(
                       1, 1, n):clone():log():sum(1):div(n):exp():squeeze()
               local bp = 1
               if input_length < target_length then
                   bp =
                       -- when evaluating (and computing BLEU at the
                       -- corpus level, we do not modify bp)
                       math.exp(1 - target_length / input_length)
               end
               self.sentence_bleu = self.sentence_bleu + curr_bleu * bp
           end
       end
       collectgarbage()
    elseif self.reward_type == 'rouge' then
        self.score[1][1] = self.score[1][1] +
            self:get_reward(target, input):sum()
        self.score[1][2] = self.score[1][2] + self:num_samples(target, input)
    else
        error('not implemented yet')
    end
end

function RewardFactory:get_corpus_score()
    if self.reward_type == 'rouge' then
        local tot_num_samples = self.score[1][2]
        local tot_recall = self.score[1][1]
        return tot_recall / tot_num_samples, tot_num_samples
    elseif self.reward_type == 'bleu' then
        local score = torch.Tensor(self.order):fill(0)
        for nn = 1, self.order do
            score[nn] = (self.score[nn][2] == 0) and 1e-16 or
                self.score[nn][1] / self.score[nn][2]
        end
        local bleu = score:log():sum(1):div(self.order):exp():squeeze()
        local bp = 1
        if self.length_input < self.length_target and self.length_input > 0 then
            bp = math.exp(1 - self.length_target / self.length_input)
        end
        print('Length of generations:', self.length_input,
              'Length of ground truth:', self.length_target,
              'Brevity penalty:', bp,
              'Corpus level BLEU:', bleu * bp,
              'Sentence level BLEU:', self.sentence_bleu / self.counter,
              'Total number of sentences:', self.counter)
        bleu = bleu * bp
        return bleu, self.counter
    else
        error('not implemented yet')
    end
end
