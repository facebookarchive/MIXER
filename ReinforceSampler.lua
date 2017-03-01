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

local ReinforceSampler, parent = torch.class('nn.ReinforceSampler',
                                              'nn.Module')
-- Module that takes a tensor storing log-probabilities (output of a LogSoftmax)
-- and samples from the corresponding multinomial distribtion.
-- Assumption: this receives input from a LogSoftMax and receives gradients from
-- a ReinforceCriterion.
function ReinforceSampler:__init(distribution)
   parent.__init(self)
   self.distribution = distribution
   self.prob = torch.Tensor()
end

function ReinforceSampler:updateOutput(input)
    if self.distribution == 'multinomial' then
        self.prob:resizeAs(input)
        self.prob:copy(input)
        self.prob:exp()
        self.output:resize(input:size(1), 1)
        if torch.typename(self.output):find('torch%.Cuda.*Tensor') then
            self.output = self.output:cudaLong()
        else
            self.output = self.output:long()
        end
        self.prob.multinomial(self.output, self.prob, 1)
        if torch.typename(self.output):find('torch%.Cuda.*Tensor') then
            self.output = self.output:cuda()
        else
            self.output = self.output:float()
        end
    else
        error('we did not implement sampling from', self.distribution)
    end
   return self.output -- batch x 1
end

-- NOTE: in order for this to work, it has to be connected
-- to a ReinforceCriterion.
function ReinforceSampler:updateGradInput(input, gradOutput)
   if self.distribution == 'multinomial' then
      -- loop over mini-batches and build sparse vector of gradients
      -- such that each sample has a vector of gradients that is all 0s
      -- except for the component corresponding to the chosen word.
      -- We assume that the gradients are provided by a ReinforceCriterion.
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      for ss = 1, self.gradInput:size(1) do
          -- adding round because sometimes multinomial returns a float 1e-6 far
          -- from an integer.
          self.gradInput[ss][torch.round(self.output[ss][1])] =
              gradOutput[ss][1]
      end
      return self.gradInput
   else
      error('we did not implement sampling from', self.distribution)
   end
end
