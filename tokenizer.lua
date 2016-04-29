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
-- Script that tokenizes the dataset and groups together samples with the
-- same source sentence length into the same bin.

require('math')
local tds = require('tds')
local pl = require('pl.import_into')()

local wordTokenizer = {}

local function cleanup_sentence(s)
    s = s:gsub("\t", "")
    -- remove leading and following white spaces
    s = s:gsub("^%s+", ""):gsub("%s+$", "")
    -- convert multiple spaces into a single space: this is needed to
    -- make the following pl.utils.split() function return only words
    -- and not white spaes
    s = s:gsub("%s+", " ")
    return s
end

function wordTokenizer.build_dictionary(filename, threshold)
    local kMaxDictSize = 5000000
    local dict = {}
    dict.symbol_to_index = {}   -- string -> id
    dict.index_to_symbol = {}   -- id -> string
    dict.index_to_freq = torch.Tensor(kMaxDictSize) -- id ->freq

    -- first add the <unk> token and the </s> token to the dictionary
    dict.symbol_to_index['<unk>'] = 1
    dict.index_to_symbol[1] = '<unk>'
    dict.index_to_freq[1] = 0
    dict.symbol_to_index['</s>'] = 2
    dict.index_to_symbol[2] = '</s>'
    dict.index_to_freq[2] = 0
    dict.separatorIndex = dict.symbol_to_index['</s>']

    -- now start counting the words
    local nr_words = 2 -- number of unique words
    local tot_nr_words = 0 -- total number of words in corpus
    local cnt = 0
    -- local inpath = paths.concat(config.root_path, filename)
    print("[ Reading from " .. filename .. ' ]')
    for s in io.lines(filename) do
        -- remove all the tabs in the string
        s = s:gsub("\t", "")
        -- convert multiple spaces into a single space: this is needed to
        -- make the following pl.utils.split() function return only words
        -- and not white spaes
        s = s:gsub("%s+", " ")
        local words = pl.utils.split(s, ' ')
        for i, word in pairs(words) do
            if word ~= "" then -- somehow the first token is always ""
                if dict.symbol_to_index[word] == nil then
                    nr_words = nr_words + 1
                    dict.symbol_to_index[word] = nr_words
                    dict.index_to_symbol[nr_words] = word
                    dict.index_to_freq[nr_words] = 1
                else
                    local indx = dict.symbol_to_index[word]
                    dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
                end
                cnt = cnt + 1
            end
        end
        -- count </s> after every line
        local indx = dict.symbol_to_index["</s>"]
        dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
        cnt = cnt + 1
    end

    dict.index_to_freq:resize(nr_words)
    tot_nr_words = dict.index_to_freq:sum()
    print("[ Done making the dictionary. ]")
    print("Training corpus statistics")
    print("Unique words: " .. nr_words)
    print("Total words " .. tot_nr_words)
    dict.tot_nr_words = tot_nr_words

    -- map rare words to special token and skip corresponding indices
    -- if the specified threshold is greater than 0
    local removed = 0
    local net_nwords = 1
    if threshold > 0 then
        for i = 2, dict.index_to_freq:size(1) do
            local word = dict.index_to_symbol[i]
            if dict.index_to_freq[i] < threshold then
                dict.index_to_freq[1] =
                    dict.index_to_freq[1] + dict.index_to_freq[i]
                dict.index_to_freq[i] = 0
                dict.symbol_to_index[word] = 1
                removed = removed + 1
            else
                -- re-adjust the indices to make them continuous
                net_nwords = net_nwords + 1
                dict.index_to_freq[net_nwords] = dict.index_to_freq[i]
                dict.symbol_to_index[word] = net_nwords
                dict.index_to_symbol[net_nwords] = word
            end
        end
        print('[ Removed ' .. removed .. ' rare words. ]')
        -- print('[ Effective number of words: ' .. net_nwords .. ' ]')
        dict.index_to_freq:resize(net_nwords)
    else
        net_nwords = nr_words
    end
    print('[ There are effectively ' .. net_nwords .. ' words in the corpus. ]')
    dict.nwords = net_nwords
    return dict
end

-- map source sentence words to id vector
local function get_source_indices(sent, dict)
    -- remove extra white spaces
    local clean_sent = cleanup_sentence(sent)
    local words = pl.utils.split(clean_sent, ' ')
    local nwords
    nwords = #words + 1
    local indices = torch.LongTensor(nwords)
    local cnt = 0
    local nsrc_unk = 0
    local unk_idx = dict.symbol_to_index['<unk>']
    local eos_idx = dict.symbol_to_index['</s>']
    for i, word in pairs(words) do
        if word ~= "" then
            local wid = dict.symbol_to_index[word]
            cnt = cnt + 1
            if wid == nil then
                indices[cnt] = unk_idx
                nsrc_unk = nsrc_unk + 1
            else
                indices[cnt] = wid
                if wid == unk_idx then
                    nsrc_unk = nsrc_unk + 1
                end
            end
        end
    end
    -- add an extra </s> at the end
    cnt = cnt + 1
    indices[cnt] = eos_idx
    return indices, indices:size(1), nsrc_unk
end

-- map target sentence words to id vector
local function get_target_indices(sent, dict, sidx)
    -- remove extra white spaces
    local clean_sent = cleanup_sentence(sent)
    local words = pl.utils.split(clean_sent, ' ')
    local nwords
    nwords = #words + 1
    local indices = torch.LongTensor(nwords, 3)
    local cnt = 1
    local ntgt_unk = 0
    local unk_idx = dict.symbol_to_index['<unk>']
    -- add </s> at the beginning of the sentence
    indices[cnt][1] = dict.symbol_to_index["</s>"]
    indices[cnt][2] = sidx
    indices[cnt][3] = cnt
    for i, word in pairs(words) do
        if word ~= "" then
            local wid = dict.symbol_to_index[word]
            if wid == nil then
                cnt = cnt + 1
                indices[cnt][1] = unk_idx
                indices[cnt][2] = sidx
                indices[cnt][3] = cnt
                ntgt_unk = ntgt_unk + 1
            else
                cnt = cnt + 1
                indices[cnt][1] = wid
                indices[cnt][2] = sidx
                indices[cnt][3] = cnt
                if wid == unk_idx then
                    ntgt_unk = ntgt_unk + 1
                end
            end
        end
    end
    return indices, indices:size(1), ntgt_unk
end


function wordTokenizer.tokenize(config, dtype, tdict, sdict, shuff)
    local tfile = paths.concat(config.root_path, config.targets[dtype])
    local sfile = paths.concat(config.root_path, config.sources[dtype])

    local tf = torch.DiskFile(tfile, 'r')
    local sf = torch.DiskFile(sfile, 'r')
    tf:quiet()
    sf:quiet()

    local source_sent_data = tds.Vec()
    local source_sent_len = {}
    local source_sent_ctr = 0
    local source_sent_nwords = 0

    local target_sent_data = tds.Vec()
    local target_sent_len = {}
    local target_sent_ctr = 0
    local target_sent_nwords = 0
    local max_target_len = 0 -- keep track of longest target sen

    local target_sen, source_sen
    target_sen = tf:readString('*l')
    source_sen = sf:readString('*l')
    while target_sen ~= '' and source_sen ~= '' do
        local tclean_sent = cleanup_sentence(target_sen)
        local twords = pl.utils.split(tclean_sent, ' ')
        local sclean_sent = cleanup_sentence(source_sen)
        local swords = pl.utils.split(sclean_sent, ' ')

        source_sent_ctr = source_sent_ctr + 1
        source_sent_data[source_sent_ctr] = sclean_sent
        target_sent_ctr = target_sent_ctr + 1
        target_sent_data[target_sent_ctr] = tclean_sent

        -- add an extra </s> at the end
        local nwords = #swords + 1
        source_sent_len[source_sent_ctr] = nwords
        source_sent_nwords = source_sent_nwords + nwords

        nwords = #twords + 1 -- add an extra </s> at the end
        target_sent_len[target_sent_ctr] = nwords
        target_sent_nwords = target_sent_nwords + nwords
        max_target_len = math.max(nwords, max_target_len)

        target_sen = tf:readString('*l')
        source_sen = sf:readString('*l')
    end
    tf:close()
    sf:close()

    assert(source_sent_ctr == target_sent_ctr)
    print('Number of sentences: ' .. target_sent_ctr)
    print('Max target sentence length: ' .. max_target_len)

    -- create the bins and their info
    local bins = {} -- each element has size, targets, sources, toffset, soffset
    bins.data = {}
    bins.nbins = 0
    -- loop over the source sentences to get bin sizes
    for i = 1, source_sent_ctr do
        local slen = source_sent_len[i]
        if bins.data[slen] == nil then
            bins.nbins = bins.nbins + 1
            bins.data[slen] = {}
            bins.data[slen].size = 1
        else
            bins.data[slen].size = bins.data[slen].size + 1
        end
    end

    -- populate the bins to store the actual source and target word indices
    for bin_dim, bin in pairs(bins.data) do
        local bin_size = bin.size
        local target_tensor_len = max_target_len * bin_size
        bin.sources = torch.LongTensor(bin_size, bin_dim):zero()
        bin.soffset = 0
        bin.targets = torch.LongTensor(target_tensor_len, 3):zero()
        bin.toffset = 1
    end

    collectgarbage()
    collectgarbage()

    local perm_vec
    -- get the permutation vector over target sentences
    if shuff == true then
        print('-- shuffling the data')
        perm_vec = torch.randperm(target_sent_ctr)
    else
        print('-- not shuffling the data')
        perm_vec = torch.range(1, target_sent_ctr)
    end

    collectgarbage()
    collectgarbage()

    print('-- Populate bins')
    -- now loop over the sentences (source and target) and populate the bins
    local nsrc_unk = 0
    local ntgt_unk = 0
    local nsrc = 0
    local ntgt = 0
    for i = 1, target_sent_ctr do
        local idx = perm_vec[i]
        if i % 10000 == 0 then
            collectgarbage()
            collectgarbage()
        end
        local curr_source_sent = source_sent_data[idx]
        local curr_target_sent = target_sent_data[idx]
        local bnum = source_sent_len[idx]
        local curr_bin = bins.data[bnum]
        curr_bin.soffset = curr_bin.soffset + 1

        local curr_source_ids, ssize, nus =
            get_source_indices(curr_source_sent, sdict)
        local curr_target_ids, tsize, nut =
            get_target_indices(curr_target_sent, tdict, curr_bin.soffset)
        nsrc = nsrc + ssize
        ntgt = ntgt + tsize
        nsrc_unk = nsrc_unk + nus
        ntgt_unk = ntgt_unk + nut

        -- load the indices into appropriate bins
        curr_bin.sources:select(1,curr_bin.soffset):copy(curr_source_ids)
        curr_bin.targets:narrow(1,curr_bin.toffset,tsize):copy(curr_target_ids)
        curr_bin.toffset = curr_bin.toffset + tsize
    end

    collectgarbage()
    collectgarbage()

    -- resize the bins.targets: yet to be done
    for bin_dim, bin in pairs(bins.data) do
        bin.targets = bin.targets:narrow(1,1,bin.toffset-1):clone()
    end

    -- finally collect all the binned source and target sentences
    local sources = {}
    local targets = {}
    for bin_dim, bin in pairs(bins.data) do
        sources[bin_dim] = bin.sources
        targets[bin_dim] = bin.targets
    end

    -- note unk rates affected by seos
    print(string.format('nlines: %d, ntokens (src: %d, tgt: %d); ' ..
                           'UNK (src: %.2f%%, tgt: %.2f%%)',
                           target_sent_ctr, nsrc, ntgt, nsrc_unk/nsrc*100,
                           ntgt_unk/ntgt*100))

    return targets, sources
end


return wordTokenizer
