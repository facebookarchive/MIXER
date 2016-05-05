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
require('nngraph')
require('xlua')
paths.dofile('ReinforceSampler.lua')
local models = {}


-- This function makes a single core element at a given time step.
function models.makeNode(args)
    local params = args.prms
    local nhid = params.n_hidden
    local input = args.inp
    local source = args.src
    local cposition = args.cpos
    local prev_h = args.prev_h
    local prev_c = args.prev_c
    local ncls_t = args.ncls_t
    local ncls_s = args.ncls_s
    local ncls_p = args.ncls_p
    local initw = args.init_w

    local function make_cuda_mm(trans_first, trans_second)
        -- Fix bug in torch.MM module with cuda.
        local mm = nn.MM(trans_first, trans_second):cuda()
        mm.gradInput[1] = mm.gradInput[1]:cuda()
        mm.gradInput[2] = mm.gradInput[2]:cuda()
        return mm
    end

    -- initialize the projection matrix
    local function initialize_proj(prj)
        prj.bias:fill(0.0)
        prj.weight:normal(0, initw)
    end

    -- component-wise addition of inp1 with linear projection of inp2, inp3
    local function new_input_sum_triple(inp1, inp2, inp3)
        local w_ih = nn.Linear(nhid, nhid)
        local w_hh = nn.Linear(nhid, nhid)
        local w_ch = nn.Linear(nhid, nhid)
        -- initialize biases to zero
        initialize_proj(w_ih)
        initialize_proj(w_hh)
        initialize_proj(w_ch)
        return nn.CAddTable()({w_ih(inp1), w_hh(inp2), w_ch(inp3)})
    end

    -- builds and returns an attention model over the source
    local function conv_attn_aux(use_cell)
        -- embedding of the source
        local nhid_c = nhid
        local src_lut = nn.LookupTable(ncls_s, nhid_c)
        src_lut.weight:normal(0, initw)
        local src_emb = src_lut(source):annotate{name = 'src_emb'}

        local pos_lut = nn.LookupTable(ncls_p, nhid_c)
        pos_lut.weight:normal(0, initw)
        local pos_emb = pos_lut(cposition):annotate{name = 'pos_emb'}

        local srcpos_emb = nn.CAddTable()(
            {src_emb, pos_emb}):annotate{name = 'srcpos_emb'}

        -- projection of previous hidden state onto source word space
        local lin_proj_hid = nn.Linear(nhid, nhid_c)
        local tgt_hid_proj = nn.View(nhid_c, 1):setNumInputDims(1)(
            lin_proj_hid(prev_h)):annotate{name = 'tgt_hid_proj'}

        local lin_proj_cel = nn.Linear(nhid, nhid_c)
        local tgt_cel_proj = nn.View(nhid_c, 1):setNumInputDims(1)(
            lin_proj_cel(prev_c)):annotate{name = 'tgt_cel_proj'}

        -- embedding of the current target word
        local tgt_lut = nn.LookupTable(ncls_t, nhid_c)
        tgt_lut.weight:normal(0, initw)
        local tgt_emb = nn.View(nhid_c, 1):setNumInputDims(1)(
            tgt_lut(input)):annotate{name = 'ttl_emb'}
        local tgt_rep = nn.CAddTable()({tgt_emb, tgt_hid_proj, tgt_cel_proj})

        local apool = params.attn_pool
        local pad = (apool - 1) / 2
        local window_model = nn.Sequential()
        window_model:add(nn.View(1, -1, nhid):setNumInputDims(2))
        window_model:add(nn.SpatialZeroPadding(0, 0, pad, pad))
        window_model:add(nn.SpatialAveragePooling(1, apool))
        window_model:add(nn.View(-1, nhid):setNumInputDims(3))
        local proc_srcpos_emb =
            window_model(srcpos_emb):annotate({name = 'proc'})

        -- distribution over source
        local scores = make_cuda_mm()(
            {proc_srcpos_emb, tgt_rep}):annotate{name = 'scores'}

        -- compute attention distribution
        local attn = nn.SoftMax()(
            nn.View(-1):setNumInputDims(2)(scores)):annotate(
            {name = 'distribution'})

        -- apply attention to the source
        local srcpos_proc = srcpos_emb
        local mout = nn.View(nhid_c):setNumInputDims(2)(
            make_cuda_mm(true, false)(
                {srcpos_proc, nn.View(-1, 1):setNumInputDims(1)(attn)}))
        return mout
    end

    local tit_lut = nn.LookupTable(ncls_t, nhid)
    tit_lut.weight:normal(0, initw)
    local inp = tit_lut(input)
    local con = conv_attn_aux(params.usecell)
    local in_gate = nn.Sigmoid()(new_input_sum_triple(inp, prev_h, con))
    local forget_gate = nn.Sigmoid()(new_input_sum_triple(inp, prev_h, con))
    local cell_gate = nn.Tanh()(new_input_sum_triple(inp, prev_h, con))
    local next_c = nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}),
                                   nn.CMulTable()({in_gate, cell_gate})})
    local out_gate = nn.Sigmoid()(new_input_sum_triple(inp, prev_h, con))
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    local w_cto = nn.Linear(nhid, nhid)
    initialize_proj(w_cto)
    local proj_ct = w_cto(con)
    local next_ct = nn.Sigmoid()(proj_ct)
    return next_h, next_c, next_ct
end

-- This function assembles the above core element
-- to build a multi-layer network at a given time step.
-- It returns the net and a table specifying the dimensions of
-- the internal states.
-- The network takes as input a table with the input, the target and
-- a table of the previous hidden states. It outputs a table
-- with the value of the loss and a table storing the hidden states
-- at the next time step.
function models.makeNetSingleStep(params, dict_target, dict_source)
    local ncls_target = dict_target.nwords
    local ncls_source = dict_source.nwords
    local ncls_position = 200
    local nhid = params.n_hidden
    local init_w = 0.05
    local prms = params
    prms.attn_pool = 5
    local dimensions_internal_states = {}
    local x = nn.Identity()()
    local previous_s = nn.Identity()()
    local next_s = {}
    local args = {prms = prms,
                  ncls_t = ncls_target,
                  ncls_s = ncls_source,
                  ncls_p = ncls_position,
                  init_w = init_w}
    local input, source, cposition = x:split(3)
    local prev_h, prev_c = previous_s:split(2)
    args.inp = input
    args.src = source
    args.cpos = cposition
    args.prev_h = prev_h
    args.prev_c = prev_c
    local next_h, next_c, ctout = models.makeNode(args)
    table.insert(next_s, next_h)
    table.insert(next_s, next_c)
    table.insert(dimensions_internal_states, nhid)
    table.insert(dimensions_internal_states, nhid)
    -- output of encoder (i.e. input of decoder) is
    -- concatenation of both long and short term memory units.
    local output_encoder = nn.JoinTable(2, 2){next_s[1], ctout}
    local num_input_decoder = nhid + nhid
    -- make the decoder
    local dec = nn.Sequential()
    dec:add(nn.Linear(num_input_decoder, ncls_target))
    dec:add(nn.LogSoftMax())
    -- init
    dec.modules[1].bias:zero()
    dec.modules[1].weight:normal(0, init_w)
    -- construct the overll network (at one time step)
    local pred = dec(output_encoder):annotate{name = 'model_prediction',
                                              description = 'output'}
    local out_sample = nn.ReinforceSampler('multinomial', false)(pred)
    local output = {nn.Identity(){out_sample, pred, output_encoder},
                    nn.Identity()(next_s)}
    local inp = {x, previous_s}
    local net = nn.gModule(inp, output)
    return net, dimensions_internal_states
end

return models
