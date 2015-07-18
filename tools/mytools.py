#!/usr/bin/env python

# ---------------------------------------
# it is a test by applying the fast rcnn
# 2015.5.7
# test some random pictures
# ---------------------------------------

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import selective_search_ijcv_with_python as selective_search



''' a dict for the net model to use '''
NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def getProposal(imageName):
    obj_proposals = selective_search.get_windows({imageName})
    return obj_proposals


def mytest(net, imageName):
    '''it is a simple test for one image'''
    # obj_proposals = getProposal(imageName)
    obj_proposals = sio.loadmat('/home/wangtw/fast-rcnn/tools/testbox.mat')['boxes']
    print("test")
    print("test")
    print(type(obj_proposals))
    print("test")
    print("test")
    print("test")
    im = cv2.imread(imageName)
    scores, boxes = im_detect(net, im, obj_proposals)

    # visualizing
    CONF_THRESH = 0.10
    NMS_THRESH = 0.3

    # change the order ?
    im = im[:, :, (2, 1, 0)]
    for cls in np.arange(len(CLASSES)):
        if cls == 0:  # not necessary back ground
            continue
        '''test the score on all the classes'''

        print(('test for{}').format(CLASSES[cls]))
        cls_boxes = boxes[:, 4 * cls: 4 * (cls + 1)]  # get boxes
        cls_scores = scores[:, cls]
        # compute the nms results
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])
                         ).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        # plot if necessary
        indexs = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(indexs) == 0:  # not necessary
            continue
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in indexs:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(CLASSES[cls], score),bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white') 

        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(CLASSES[cls], CLASSES[cls], CONF_THRESH), fontsize=14)

        plt.axis('off')
        plt.tight_layout()
        plt.draw()


def parse_args():
    """Parse input arguments.
    note that the input image must be specific"""

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='[vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--image', dest='imageName', help='testImage',
                        default='\home\wtw\123.jpg', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    prototxt = os.path.join('models', NETS[args.demo_net][0],
                              'test.prototxt')
    caffemodel = os.path.join('data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError('no caffemodel')
    if not os.path.isfile(prototxt):
        raise IOError('no prototxt')

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    mytest(net, args.imageName)
    plt.show()
