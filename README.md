# MIXER - Sequence Level Training with Recurrent Neural Networks
http://arxiv.org/abs/1511.06732

This is a self contained software accompanying the paper titled: Sequence Level
Training with Recurrent Neural Networks: http://arxiv.org/abs/1511.06732.
The code allows you to reproduce our result on the machine translation task.

The code implements MIXER; it runs both training and evaluation.

## Preparing the training data
run prepareData.sh

## Examples
Here are some examples of how to use the code.

* To run an LSTM with the default parameter setting used to generate MIXER's entry for machine translation (see table in fig.5 of http://arxiv.org/abs/1511.06732), type
```
th -i main.lua
```

* To run an LSTM with following
hyper-parameters:
** hidden units: 128
** minibatch size: 64
** learning rate: 0.1
** number of time steps we unfold: 15
type
```
th -i main.lua -nhid 128 -bsz 64 -lr 0.1 -bptt 15
```

To list all the options available, you need to type
```
th main.lua --help
```

## Requirements
The software is written in Lua. It requires the following packages:
* Torch 7
* nngraph
* cutorch
* cunn
It runs on standard Linux box with GPU.

## Installing
Download the files in an appropriate directory and run the code from there. See below.


## How it works
The top level file is called main.lua. In order to run the code
you need to run the file using torch. For example:
```
th -i main.lua -<option1_name> option1_val -<option2_name> option2_val ...
```

## Structure of the code.
* main.lua this is the scripts that launches training and testing. The user can pass options to set various hyper-parameters, such as learning rate, number of hidden units, etc.
* Trainer.lua  this is a simple class that loops over the dataset a certain number of epochs to train the model, that loops over the validation/test set to evaluate and that backups.
* model_factory.lua  this is a function which returns the network operating at a single time step.
* Mixer.lua  this is the class which implements the unrolled recurrent network, cloning as many steps as necessary whatever is returned by model_factory. It implements the basic the basic fprop/bprop through the recurrent model.
* ReinforceSampler.lua  class that is used to sample from a tensor storing log-probabilities.
* ReinforceCriterion.lua  criterion which is used to compute reward once the end of sequence is reached.
* ClassNLLCriterionWeighted.lua  wrapper around ClassNLLCriterion which multiplies the output of ClassNLLCriterion by a scalar to weigh the loss.
* LinearNoBackpropInput.lua  just like Linear but without computing derivatives w.r.t. input.
* DataSource.lua  data provider that takes as input a tokenized dataset in binary format and returnes mini-batches.
* reward_factory.lua  class that is used to compute BLEU and ROUGE scores (both at the sentence and corpus level).
* util.lua  auxiliary functions.

## License
"MIXER"'s software is BSD-licensed.
We also provide an additional patent grant.


## Other Details
See the CONTRIBUTING file for how to help out.
