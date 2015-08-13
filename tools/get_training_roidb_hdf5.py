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
import roi_data_layer.roidb as rdl_roidb
import h5py
import cv2


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
                        default='1', type=int)
    parser.add_argument('--output', dest='output_dir',
                        help='dataset to train on',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
    
def get_im_blob_in_one_function(image_path, roi_boxes, image_flipped_flag):
    # the first phase, we reshape the image and calculate the shift value
    im = cv2.imread(image_path)
    im = im.astype(np.float, copy=False)
    im -= cfg.PIXEL_MEANS  # subtract the image mean
    if image_flipped_flag:
        im = im[:, ::-1, :]

    # transpose the image into the BGR order
    channel_swap = (2, 0, 1)
    im = im.transpose(channel_swap)

    width = im.shape[0]
    height = im.shape[1]
    assert cfg.HDF5_IMAGE_HEIGHT == cfg.HDF5_IMAGE_WIDTH, \
            "The function of differert width and height is not supported yet!"
    if width > height:
        # we resize the image width to the cfg
        output_width = cfg.HDF5_IMAGE_WIDTH
        output_height =  int(float(height) / width * cfg.HDF5_IMAGE_HEIGHT)
        shift_x = 0
        shift_y = int(float(cfg.HDF5_IMAGE_HEIGHT - output_height) / 2)
    else:
        output_height = cfg.HDF5_IMAGE_HEIGHT
        output_width = int(float(width) / height * cfg.HDF5_IMAGE_WIDTH)
        shift_y = 0
        shift_x = int(float(cfg.HDF5_IMAGE_WIDTH - output_width) / 2)
    
    im = cv2.resize(im, output_width, output_height, 
        interpolation=cv2.INTER_LINEAR)  # resize the image 
    stuffed_image = np.zeros((cfg.HDF5_IMAGE_WIDTH, cfg.HDF5_IMAGE_HEIGHT, 3),
        dtype=np.float32)
    # copy the image data into the stuffed_image
    stuffed_image[shift_x:, shift_y:, :] = im.copy()

    # now for the rois! 1,3 -> witdh; 2,4->height
    roi_boxes[:, 1] = roi_boxes[:, 1] + shift_x
    roi_boxes[:, 3] = roi_boxes[:, 3] + shift_x
    roi_boxes[:, 2] = roi_boxes[:, 2] + shift_y
    roi_boxes[:, 4] = roi_boxes[:, 4] + shift_y
    
    return im, roi_boxes


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
    
    if args.output_dir == '':
        output_dir = os.path.abspath(os.path.join(ROOT_DIR, 'data', 'hdf5'))
    else:
        output_dir = args.output_dir
    
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
    bbox_means, bbox_stds = \
        rdl_roidb.add_bbox_regression_targets(roidb)    
    # get the index of the fetching
    for i_round in xrange(args.number_of_round):    
        index = np.random.permutation(np.arange(len(roidb)))
        cur = 0
        
        # the multi_label_softmax, we store the below datasets
        # 1. data blob, 
        file_image = h5py.File(os.path.join(output_dir, 'image' + \
            '_' + str(i_round) +'.h5'), 'w')
        image_dset = file_image.create_dataset("data", (1, 3, 600, 600), 
            maxshape=(None, 3, 600, 600), dtype='float32')
   
         # 2. multi_label blob
        file_multilabel = h5py.File(os.path.join(output_dir, 'multi_label' + \
            '_' + str(i_round) +'.h5'), 'w')
        sleeve_dset = file_multilabel.create_dataset("sleeve", (64, 1, 1, 1), 
            maxshape=(None, 1, 1, 1), dtype='float32')
        texture_dset = file_multilabel.create_dataset("texture", (64, 1, 1, 1), 
            maxshape=(None, 1, 1, 1), dtype='float32')
        neckband_dset = file_multilabel.create_dataset("neckband", (64, 1, 1, 1), 
            maxshape=(None, 1, 1, 1), dtype='float32')
                                         
        # 3. rois blob, 
        file_rois = h5py.File(os.path.join(output_dir, 'rois' + \
            '_' + str(i_round) +'.h5'), 'w')
        rois_dset = file_rois.create_dataset("rois", (64, 5, 1, 1), 
            maxshape=(None, 5, 1, 1), dtype='float32')
        
        # 4. bbox_tartgets, bbox_loss_weight
        file_bbox = h5py.File(os.path.join(output_dir, 'bbox' + \
            '_' + str(i_round) +'.h5'), 'w')
        bbox_targets_dset = \
            file_bbox.create_dataset("bbox_targets", (64, 12, 1, 1), 
            maxshape=(None, 12, 1, 1), dtype='float32')
        bbox_loss_weights_dset = \
            file_bbox.create_dataset("bbox_loss_weights", (64, 12, 1, 1), 
                maxshape=(None, 12, 1, 1), dtype='float32')
        
        # 5. class label
        file_class_label = h5py.File(os.path.join(output_dir, 'class_label' + \
            '_' + str(i_round) +'.h5'), 'w')
        image_dset = file_class_label.create_dataset("labels", (64, 1, 1, 1), 
            maxshape=(None, 1, 1, 1), dtype='float32')
            
        while cur <= len(index):
            # do the sampling work
            db_inds = index[cur: cur + cfg.TRAIN.IMS_PER_BATCH]            
            
            cur_db = [roidb[i] for i in db_inds]
            
            blob = get_minibatch(cur_db, cfg.HDF5_NUM_CLASS, cfg.HDF5_NUM_LABEL)

            for i_image in xrange(cfg.TRAIN.IMS_PER_BATCH):
                # now write the dataset image by image
                # 2. multi_label blob                
                sleeve_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['sleeve'].reshape(128, 1, 1, 1)
                texture_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['texture'].reshape(128, 1, 1, 1)
                neckband_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['neckband'].reshape(128, 1, 1, 1)
                    
                # 4. bbox_tartgets, bbox_loss_weight
                bbox_targets_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['bbox_targets'].reshape(128, 12, 1, 1)
                bbox_loss_weights_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['bbox_loss_weights'].reshape(128, 12, 1, 1)
                
                image_dset[cur * 128 : cur * 128 + 128, :, :, :] = \
                    blob['labels'].reshape(128, 1, 1, 1)
                    
                if cfg.HDF5_BYPASS_SYS_IM_ROIS:            
                    image_blob = get_im_blob_in_one_function(image_path, 
                        roi_boxes, image_flipped_flag)
            
                # get the image data blob and the image rois blob
            
            cur = cur + cfg.TRAIN.IMS_PER_BATCH
    
