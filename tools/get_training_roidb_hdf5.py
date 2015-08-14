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
import timeit

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

    # image is already the BGR order
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
    scale_width = float(output_width) / float(width)
    scale_height = float(output_height) / float(height)
    im = cv2.resize(im, dsize=(output_height, output_width), 
        interpolation=cv2.INTER_LINEAR)  # resize the image 
    stuffed_image = np.zeros((cfg.HDF5_IMAGE_WIDTH, cfg.HDF5_IMAGE_HEIGHT, 3),
        dtype=np.float32)
    # copy the image data into the stuffed_image
    if stuffed_image[shift_x : shift_x + im.shape[0],
                  shift_y : shift_y + im.shape[1], :].shape[0] != im[0 : 0 + im.shape[0], 
                     0 : 0 + im.shape[1], :].shape[0] or stuffed_image[shift_x : shift_x + im.shape[0],
                  shift_y : shift_y + im.shape[1], :].shape[1] != im[0 : 0 + im.shape[0], 
                     0 : 0 + im.shape[1], :].shape[1]:
        print 'test'
    stuffed_image[shift_x : shift_x + im.shape[0],
                  shift_y : shift_y + im.shape[1], :] = \
                  im[0 : 0 + im.shape[0], 
                     0 : 0 + im.shape[1], :]

    # transpose the im into the (3,600,600)
    channel_swap = (2, 0, 1)
    stuffed_image = stuffed_image.transpose(channel_swap)

    # now for the rois! 1,3 -> witdh; 2,4->height
    roi_boxes[:, 1] = roi_boxes[:, 1] * scale_width + shift_x
    roi_boxes[:, 3] = roi_boxes[:, 3] * scale_width + shift_x
    roi_boxes[:, 2] = roi_boxes[:, 2] * scale_height + shift_y
    roi_boxes[:, 4] = roi_boxes[:, 4] * scale_height + shift_y
    
    return stuffed_image.astype(np.uint8), roi_boxes


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
    if not 'roidb' in globals():
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                origin_roidb = cPickle.load(fid)
                print('The precomputed roidb datasets loaded')
                bbox_means, bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(origin_roidb)    
        else:
            imdb = get_imdb(args.imdb_name)
            print('No cache file spotted. Making one from the scratch')
            print('Loaded dataset `{:s}`'.format(imdb.name))
            origin_roidb = get_training_roidb(imdb)
            
            with open(cache_file, 'wb') as fid:
                cPickle.dump(origin_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print('The precomputed roidb saved to {}'.format(cache_file))
            bbox_means, bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(origin_roidb)    
    print('Generating the hdf5 training data')
    # generate the training label datasets
    
    # get the index of the fetching
    # devide the dataset into 4 part
    part_num = 17
    for i_round in xrange(args.number_of_round):
        for i_part in xrange(part_num):
            step = int(len(origin_roidb) / part_num)
            roidb = origin_roidb[i_part * step : (i_part + 1) * step]
        
            index = np.random.permutation(np.arange(len(roidb)))
            cur = 0
            
            # the multi_label_softmax, we store the below datasets
            # 1. data blob, 
            file_image = h5py.File(os.path.join(output_dir, 'image' + \
                '_' + str(i_round).zfill(2) +'_'+ str(i_part).zfill(2) + '.h5'), 'w')
            image_data = np.zeros(((1 * len(roidb), 3, cfg.HDF5_IMAGE_WIDTH, 
                                    cfg.HDF5_IMAGE_HEIGHT)), dtype=np.uint8)
       
             # 2. multi_label blob
            file_multilabel = h5py.File(os.path.join(output_dir, 'multi_label' + \
                '_' + str(i_round).zfill(2) +'_'+ str(i_part).zfill(2) + '.h5'), 'w')
            neckband_data = np.zeros(((64 * len(roidb), 1, 1, 1)), dtype=np.float32)
            texture_data = np.zeros(((64 * len(roidb), 1, 1, 1)), dtype=np.float32)
            sleeve_data = np.zeros(((64 * len(roidb), 1, 1, 1)), dtype=np.float32)
                                             
            # 3. rois blob, 
            file_rois = h5py.File(os.path.join(output_dir, 'rois' + \
                '_' + str(i_round).zfill(2) +'_'+ str(i_part).zfill(2) + '.h5'), 'w')
            rois_data = np.zeros(((64 * len(roidb), 5, 1, 1)), dtype=np.float32)
            
            # 4. bbox_tartgets, bbox_loss_weight
            file_bbox = h5py.File(os.path.join(output_dir, 'bbox' + \
                '_' + str(i_round).zfill(2) +'_'+ str(i_part).zfill(2) + '.h5'), 'w')
            bbox_targets_data = np.zeros(((64 * len(roidb), 16, 1, 1)), dtype=np.float32)
            bbox_loss_weights_data = np.zeros(((64 * len(roidb), 16, 1, 1)), dtype=np.float32)
            
            # 5. class label
            file_class_label = h5py.File(os.path.join(output_dir, 'class_label' + \
                '_' + str(i_round).zfill(2) +'_'+ str(i_part).zfill(2) + '.h5'), 'w')
            image_label_data = np.zeros(((64 * len(roidb), 1, 1, 1)), dtype=np.float32)
            
            tic = timeit.default_timer()
            while cur < len(index):
                # do the sampling work
                if cur % 100 == 2:
                    print('Processing the {} th image data'.format(cur))
                    toc = timeit.default_timer()
                    
                    print('Time spent on processing 100 image is {}'.format(toc - tic))
                    tic = timeit.default_timer()
                    
                        
                db_inds = index[cur: cur + cfg.TRAIN.IMS_PER_BATCH]            
                
                cur_db = [roidb[i] for i in db_inds]
                
                blob = get_minibatch(cur_db, cfg.HDF5_NUM_CLASS + 1, cfg.HDF5_NUM_LABEL)
    
                # 2. multi_label blob               
                sleeve_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['sleeve'].reshape(cfg.TRAIN.BATCH_SIZE, 1, 1, 1)
                texture_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['texture'].reshape(cfg.TRAIN.BATCH_SIZE, 1, 1, 1)
                neckband_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['neckband'].reshape(cfg.TRAIN.BATCH_SIZE, 1, 1, 1)
                        
                # 4. bbox_tartgets, bbox_loss_weight
                bbox_targets_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['bbox_targets'].reshape(cfg.TRAIN.BATCH_SIZE, 16, 1, 1)
                bbox_loss_weights_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['bbox_loss_weights'].reshape(cfg.TRAIN.BATCH_SIZE, 16, 1, 1)
                    
                # 5. class label
                image_label_data[cur * cfg.TRAIN.BATCH_SIZE / 2: \
                        (cur + 2) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                        blob['labels'].reshape(cfg.TRAIN.BATCH_SIZE, 1, 1, 1)
                
                for i_image in xrange(cfg.TRAIN.IMS_PER_BATCH):
                    # now write the dataset image by image
                        
                    if cfg.HDF5_BYPASS_SYS_IM_ROIS:
                        image_blob, rois_blob = \
                            get_im_blob_in_one_function(\
                            os.path.abspath(blob['data'][i_image]), 
                            blob['rois'][i_image * cfg.TRAIN.BATCH_SIZE / 2: \
                            (i_image + 1)* cfg.TRAIN.BATCH_SIZE / 2], 
                            cur_db[i_image]['flipped'])
                        # the image is 600 * 600 * 3, now we need a 
                        image_data[cur + i_image, :, :, :] = \
                            image_blob.reshape(1, 3, 
                            cfg.HDF5_IMAGE_WIDTH, cfg.HDF5_IMAGE_HEIGHT)
                        rois_data[(cur + i_image) * cfg.TRAIN.BATCH_SIZE / 2 : \
                            (cur + i_image + 1) * cfg.TRAIN.BATCH_SIZE / 2, :, :, :] = \
                            rois_blob.reshape(cfg.TRAIN.BATCH_SIZE / 2, 5, 1, 1)
                
                    # get the image data blob and the image rois blob
                
                cur = cur + cfg.TRAIN.IMS_PER_BATCH
            
            
            sleeve_dset = file_multilabel.create_dataset("sleeve", data=sleeve_data)
            texture_dset = file_multilabel.create_dataset("texture", data=texture_data)
            neckband_dset = file_multilabel.create_dataset("neckband", data=neckband_data)    
            image_dset = file_image.create_dataset("data", data=image_data)            
            image_label_dset = file_class_label.create_dataset("labels", data=image_label_data)
            bbox_targets_dset = \
                file_bbox.create_dataset("bbox_targets", data=bbox_targets_data)
            bbox_loss_weights_dset = \
                file_bbox.create_dataset("bbox_loss_weights", data=bbox_loss_weights_data)
            rois_dset = file_rois.create_dataset("rois", data=rois_data)
      
            file_class_label.close()
            file_bbox.close()
            file_multilabel.close()
            file_image.close()
            file_rois.close()
        