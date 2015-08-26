#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import caffe
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
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    
    if cfg.SEP_DETECTOR:
        cache_path = os.path.abspath(os.path.join(ROOT_DIR, 'data', 'cache'))
        cache_file = os.path.join(cache_path, \
            args.imdb_name + '_3CL=' + str(cfg.ThreeClass) + \
            '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
            '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
            '_BLC=' + str(cfg.BALANCED) + \
            '_COF=' + str(cfg.BALANCED_COF) + \
            '_TT1000=' + str(cfg.TESTTYPE1000) + \
            '_SEP_DETECTOR' + str(cfg.SEP_DETECTOR_NUM) + \
            '_solver_roidb.pkl')
    else:
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
            print('The precomputed roidb loaded')
            output_dir = get_output_dir(args.imdb_name, None)
    else:    
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        roidb = get_training_roidb(imdb)
        output_dir = get_output_dir(imdb.name, None)
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'The precomputed roidb saved to {}'.format(cache_file)
        
    #//output_dir = output_dir + '_test'
    print 'Output will be saved to `{:s}`'.format(output_dir)
    #if cfg.SEP_DETECTOR:
        # in this case we need to change the class if not
        #output_dir = output_dir + str(cfg.SEP_DETECTOR_NUM)
        #for i_roidb in xrange(len(roidb)):
            #if roidb[i_roidb]['gt_classes'][0] != cfg.SEP_DETECTOR_NUM:
                
                
            # process the image one by one
    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
