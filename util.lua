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
require('math')
local utils = {}

-----------------------------------------------------------------
-- util functions useful to compute BLEU/ROUGE metrics (cpu-based).

-- the first n entry (integers) of the input vector vec
-- are hashed to a unique integer;
-- e.g. n=3, (i, j, k) -> (i-1)*V^2 + (j-1)*V + (k - 1) + 1
-- V is the vocabulary size
function utils.compute_hash(vec, n, V)
   assert(vec:size(1) >= n)
   local hash = 0
   for cnt = 1, n do
      hash = hash + (vec[cnt] - 1) * math.pow(V, n - cnt)
   end
   return hash + 1 -- we start counting from one
end

-- compute ngram counts
-- input is a 1D tensor storing the indexes of the words in the sequence.
-- if skip_id is not nil, then the ngram is skipped.
function utils.get_counts(input, nn, V, skip_id, output)
   local sequence_length = input:size(1)
   assert(nn <= sequence_length)
   local out = (output == nil) and {} or output
   for tt = 1, sequence_length - nn + 1 do
       local curr_window = input:narrow(1, tt, nn)
       -- add to hash table only if we do not skip, or we skip but there
       -- is no skip_id in the current window.
       -- This is used to skip UNK tokens from the reference.
       if skip_id == nil or curr_window:eq(skip_id):sum() == 0 then
           local hash = utils.compute_hash(
               curr_window, nn, V)
           if out[hash] == nil then
               out[hash] = 1
           else
               out[hash] = out[hash] + 1
           end
       end
   end
   return out
end

-- compute partial bleu score given counts
function utils.compute_score(counts_input, counts_target, smoothing_val)
   local tot = 0
   local score = 0
   for k, v in pairs(counts_input) do
      tot = tot + v
      if counts_target[k] ~= nil then
         if counts_input[k] > counts_target[k] then
            score = score + counts_target[k]
         else
            score = score + counts_input[k]
         end
      end
   end
   tot = tot + smoothing_val
   score = score + smoothing_val
   score = (tot > 0) and score / tot or 0
   return score
end

function utils.compute_precision(score, input, target, smoothing_val)
   local tot = 0
   local prec = 0
   for k, v in pairs(input) do
      tot = tot + v
      if target[k] ~= nil then
          if input[k] > target[k] then
              prec = prec + target[k]
         else
             prec = prec + input[k]
         end
      end
   end
   score[1] = score[1] + prec
   score[2] = score[2] + tot
   -- This is used for sentence level BLEU, which is always smoothed
   score[3] = (prec + smoothing_val) / (tot + smoothing_val)
end

function utils.compute_recall(input, target)
   local tot = 0
   local matches = 0
   for k, v in pairs(target) do
      tot = tot + v
      if input[k] ~= nil then
          matches = matches + math.min(input[k], target[k])
      end
   end
   return (tot == 0) and 0 or matches / tot
end

------------------------------------------------------
-- util functions used by the model iself
function utils.scale_clip(dat, th)
    -- global normalization of the whole tensor;
    -- use this when normalizing the (gradients of the) parameters,
    -- for instance.
    local dat_norm = dat:norm()
    if dat_norm > th then
        dat:div(dat_norm/th)
    end
end


return utils
