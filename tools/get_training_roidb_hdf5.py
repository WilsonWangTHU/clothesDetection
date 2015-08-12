#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
from roi_data_layer.minibatch import get_minibatch
import argparse
from datasets import ROOT_DIR
import pprint
import numpy as np
import sys
import os
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--round', dest='number_of_round',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Starting to generate the fast-rcnn training roidb hdf5')

    print('Called with args:')
    print(args)


    print('Using config:')
    pprint.pprint(cfg)

    
    cache_path = os.path.abspath(os.path.join(ROOT_DIR, 'data', 'cache'))
    cache_file = os.path.join(cache_path, \
        args.imdb_name + '_3CL=' + str(cfg.ThreeClass) + \
        '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
        '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
        '_BLC=' + str(cfg.BALANCED) + \
        '_COF=' + str(cfg.BALANCED_COF) + \
        '_TT1000=' + str(cfg.TESTTYPE1000) + \
        '_solver_roidb.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
            print('The precomputed roidb datasets loaded')
    else:
        imdb = get_imdb(args.imdb_name)
        print('No cache file spotted. Making one from the scratch')
        print('Loaded dataset `{:s}`'.format(imdb.name))
        roidb = get_training_roidb(imdb)
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('The precomputed roidb saved to {}'.format(cache_file))
    
    print('Generating the hdf5 training data')
    # generate the training label datasets
    
    # get the index of the fetching
    for i in xrange(args.number_of_round):    
        index = np.random.permutation(np.arange(len(roidb)))
        cur = 0
        while cur <= len(index):
            # do the sampling work
            db_inds = index[cur: cur + cfg.TRAIN.IMS_PER_BATCH]            
            
            cur_db = [roidb[i] for i in db_inds]
            
            blob = get_minibatch(cur_db, cfg.HDF5_NUM_CLASS, cfg.HDF5_NUM_LABEL)
            
            cur = cur + cfg.TRAIN.IMS_PER_BATCH
        
        
    
    
    
    
    
    
    
    
    