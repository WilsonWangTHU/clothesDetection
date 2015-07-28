#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import struct

CONF_THRESH = 0.5
NMS_THRESH = 0.3
font = cv2.FONT_HERSHEY_SIMPLEX

CLASSES = ('__background__',
           'Upper', 'Lower', 'Whole')

def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    dets = dets[inds, :]
    
    if len(inds) == 0:
        print "no detection!!!!! for class " + class_name
        cv2.imwrite("/home/twwang/demo_results/" + \
            os.path.split(image_name)[1], im)
        return
    else:
        cv2.rectangle(im,(dets[0, 0], dets[0, 1]),
                        (dets[0, 2], dets[0, 3]),
                        (0,255,0), 3)
        cv2.putText(im, 'Cls: ' + class_name + ',P: ' + str(dets[0, 4]),
            (dets[0, 0], dets[0, 1]),
            font, 0.5, (0,255,0), 1, 255)
        #print im.shape
        cv2.imwrite("/home/twwang/demo_results/" + \
            os.path.split(image_name)[1], im)

def demo(net, image_name, classes):

    # get the proposals by using the shell to use c++ codes    
    os.system(
        '/media/Elements/twwang/fast-rcnn/rcnn_test/proposals_for_python.sh' \
        + ' ' + image_name)
    
    # Load computed Selected Search object proposals
    data = open('/home/twwang/temp_proposal', "rb").read()
    number_proposals = struct.unpack("i", data[0:4])[0]
    number_edge = struct.unpack("i", data[4:8])[0]
    
    number_proposals = min(cfg.NUM_PPS, number_proposals)
    obj_proposals = np.asarray(struct.unpack(
        str(number_proposals * 4) + 'f',
        data[8: 8 + 16 * number_proposals])).reshape(number_proposals, 4)
    
    #box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
    #                        image_name + "04" + '_boxes.mat')
    #obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im = cv2.imread(image_name)    
    #print im.shape
    im = cv2.flip(im, 0)
    im = cv2.transpose(im)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class

    for cls in ['Upper', 'Lower', 'Whole']:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')    
    parser.add_argument('--image', dest='img',
                        help='The input image',)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'ClothCaffeNet',
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR+
    '/output/default/clothesDataset_3CL=True_BLC=True_COF=True_TT1000=True',
    'caffenet_fast_rcnn_iter_40000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print('Processing the image')
    demo(net, args.img, ('Upper',))