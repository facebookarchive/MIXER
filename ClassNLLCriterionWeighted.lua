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
-- This class is like NLL, except that it also takes a global
-- weighting scalar, by which we scale both the output loss
-- and the gradients w.r.t. input
local ClassNLLCriterionWeighted, parent = torch.class(
   'nn.ClassNLLCriterionWeighted', 'nn.ClassNLLCriterion')

function ClassNLLCriterionWeighted:__init(globalWeight, weights, sizeAverage)
   parent.__init(self, weights, sizeAverage)
   self.globalWeight = globalWeight or 1
end

function ClassNLLCriterionWeighted:updateOutput(input, target)
   local result1, result2 = parent.updateOutput(self, input, target)
   return self.globalWeight * result1, result2
end

function ClassNLLCriterionWeighted:updateGradInput(input, target)
   local result1, result2 = parent.updateGradInput(self, input, target)
   return result1:mul(self.globalWeight), result2
end
