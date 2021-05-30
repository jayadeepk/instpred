# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""This script is used to precompute features extracted by the Mask R-CNN
backbone."""

import os
import pprint
import copy
import numpy as np
import pickle
from detectron.utils.io import save_object

from mypython.logger import create_logger
from npy_append_array import NpyAppendArray

# Save results every SAVE_IT iterations
SAVE_IT = 50

#-------------------------------------------------------------------------------
# Load config
from config import make_config
opt, configs, _ = make_config()
assert opt['precompute_features'], 'please specify which types of features to compute, ex : fpn_res5_2_sum'

#-------------------------------------------------------------------------------
# Start logging
logger = create_logger(os.path.join(opt['logs'], 'compute_features.log'))
logger.info('============ Initialized logger ============')

#-------------------------------------------------------------------------------
# Create dataloaders
trainsetConfig = configs['trainset']
valsetConfig = configs['valset']

from data import load_cityscapes_train, load_cityscapes_val
logger.info(pprint.pprint(trainsetConfig))
trainset, loaded_model = load_cityscapes_train(trainsetConfig)
# to remove shuffling of the dataset
trainset.data_source.dataset.dataset = trainset.data_source.dataset.dataset.dataset
trainLoader = iter(trainset)

valsetConfig['loaded_model'] = loaded_model
logger.info(pprint.pprint(valsetConfig))
valset = load_cityscapes_val(valsetConfig)
valLoader = iter(valset)

def pickle_load(path):
    if not os.path.isfile(path):
        return []
    with open(path, 'rb') as f:
        a = pickle.load(f)['sequence_ids']
    return a

def pickle_save(seq_ids, path):
    with open(path, 'wb') as f:
        pickle.dump(dict(sequence_ids=seq_ids), f, protocol=2)

#-------------------------------------------------------------------------------
# Precompute features

def precompute_maskrcnn_backbone_features(config, dataset, split):
    feat_type = config['feat_type']
    # assert that the dimensions are ok otherwise break
    nI, nT = config['n_input_frames'], config['n_target_frames']

    # Automatically get feature dimensions
    sample_input, _, _ = dataset.next()
    sample_features = sample_input[feat_type]
    assert sample_features.dim() == 4, "Batch mode is expected"
    sz = sample_features.size()
    assert(sz[0] == 1, 'This function assumes batch mode, but a single example per batch')
    c, h, w = sz[1]//nI, sz[2], sz[3]
    assert c == config['nb_features']
    dataset.reset()
    # Check that the dataset to compute will be under 100GB - floating point takes 4B - check
    assert len(dataset) * (nI+nT) * c * h * w * 4 <= 1e11, \
        'The dataset to compute will take over 100 GB - aborting'

    filename = os.path.join(opt['save'] , '__'.join(
        (split, feat_type, 'nSeq%d'%(nI+nT), 'fr%d'%config['frame_ss'])))
    logger.info('Results folder: %s'%filename)
    if os.path.isfile(filename + '__features.npy'):
        arr = np.load(filename + '__features.npy', mmap_mode='r')
        logger.info('Loading previous results. Shape: ' + str(arr.shape))
        # Get iteration number to resume from
        resume_it = len(arr)
        del arr
    else:
        resume_it = 0
    # Open output numpy file in append mode
    output_numpy_file = NpyAppendArray(filename + '__features.npy')

    # Initialize tensors

    seq_features = np.empty((0, (nI+nT), c, h, w), dtype = np.float32)
    seq_ids = pickle_load(filename + '__ids.pkl')
    dataset.current = resume_it
    for i, data in enumerate(dataset):
        logger.info(i + resume_it)
        inputs, targets, _ = data
        correspondingSample = dataset.data_source.dataset.dataset.im_list[i + resume_it]['image']
        # insert in the dataset
        seq_ids.append(correspondingSample)
        inp_feat = inputs[feat_type].view((nI, c, h, w)).numpy().astype(np.float32)
        tar_feat = targets[feat_type].view((nT, c, h, w)).numpy().astype(np.float32)
        feat = np.concatenate((inp_feat, tar_feat), axis=0)
        feat = np.expand_dims(feat, axis=0)
        seq_features = np.append(seq_features, feat, axis=0)
        # Append results every SAVE_IT iterations, because keeping
        # large numpy files in memory might invoke Out-Of-Memory Killer.
        if (i + 1) % SAVE_IT == 0:
            pickle_save(seq_ids, filename + '__ids.pkl')
            output_numpy_file.append(seq_features)
            seq_features = np.empty((0, (nI+nT), c, h, w), dtype = np.float32)
        if i >= (config['it']-1) :
            break
    if seq_features.size:
        output_numpy_file.append(seq_features)
    pickle_save(seq_ids, filename + '__ids.pkl')
    logger.info('Precomputed features saved to : %s'%filename)


assert opt['trainbatchsize'] == opt['valbatchsize'] == 1
cfg_train = {
    'n_input_frames' : opt['n_input_frames'],
    'n_target_frames' : opt['n_target_frames'],
    'nb_features' : opt['nb_features'],
    'feat_type' : opt['precompute_features'],
    'frame_ss' : opt['frame_subsampling'],
    'it' : opt['ntrainIt']
}
cfg_val = copy.copy(cfg_train)
cfg_val['it'] = opt['nvalIt']
print(cfg_train)
print(cfg_val)

# precompute_maskrcnn_backbone_features(cfg_val, valLoader, 'val')
precompute_maskrcnn_backbone_features(cfg_train, trainLoader, 'train')
