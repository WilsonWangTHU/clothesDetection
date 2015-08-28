# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:58:35 2015

@author: twwang
"""

#coding=utf-8
# -------------------------------------------------------
#   it is a python port of network test, generate the 
#   results of the three class: Upper body(1), Lower 
#   body (2), and the whole body(3)
#
#   Written by Tingwu Wang, 17.7.2015
# -------------------------------------------------------

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import numpy as np
import cPickle

import caffe, os, sys, cv2
import argparse
import struct
import datetime

texture_classes = ('一致色', '横条纹', '纵条纹',
        '其他条纹', '豹纹斑马纹', '格子',
        '圆点', '乱花', 'LOGO及印花图案', '其他'
        )
neckband_classes = ('圆领', 'V领', '翻领',
        '立领', '高领', '围巾领',
        '一字领', '大翻领西装领',
        '连帽领', '其他'
        )
sleeve_classes = ('短袖', '中袖', '长袖')

texture_classes = ('single_color', 'horizon_strip', 'vertical_strip',
        'leopard_or_zebra', 'grid',
        'dot', 'random_match', 'LOGO', 'other'
        )
neckband_classes = ('round_collar', 'V_collar', 'turn_down_collar',
        'stand_collar', 'high_collar', 'shawl_collar',
        'horizon_collar', 'golila_or_tailored',
        'hooded_colar', 'other_collar'
        )
sleeve_classes = ('short_sleeve', 'middle_sleeve', 'long_sleeve')

texture_to_label_ind = dict(zip(
    xrange(len(texture_classes)), texture_classes))
neckband_to_label_ind = dict(zip(
    xrange(len(neckband_classes)),
    neckband_classes
    ))
sleeve_to_label_ind = dict(zip(
    xrange(len(sleeve_classes)),
    sleeve_classes   
    ))

TYPE_MAPPER = {1 : [1,2,3,4,5,6,7,9,10,12,13,14,15,16,17,18,19], 
               2 : [21,22,23,24,25,26], 
               3 : [8, 11, 20]}

font = cv2.FONT_HERSHEY_SIMPLEX

num_category = 26
num_class = 3

top_number = 10

pred_preCls = np.zeros((26, 26), dtype=np.float)

CONF_THRESH = 0.1
PLOT_CONF_THRESH = 0.6
PLOT_MULTI_LABEL_THRESH = 0.4
NMS_THRESH = 0.3

subcategories = dict()

subcategories[1] = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 
    'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 
    'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 
    'Tee', 'Top', 'Turtleneck'];

subcategories[2] = ['Capris', 'Chinos', 'Culottes', 'Cutoffs', 
   'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 
   'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks'];

subcategories[3] = ['Caftan', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 
    'Kaftan', 'Kimono', 'Onesie', 'Robe', 'Romper'];


CLASSES = ('__background__', 'Upper', 'Lower', 'Whole')

# it is the error file, we record the error files here
error_parsing_file = None

def google_top(net, input_path, sub_file, output_path):
    assert cfg.MULTI_LABEL == False, 'The multilabel function not implemented'

    if not os.path.exists(os.path.join(output_path, 'images', sub_file)):
        os.mkdir(os.path.join(output_path, 'images', sub_file))
    
    result_file = open(os.path.join(output_path, 'txt', sub_file + '.txt'), 'w')
    # Load image one by one
    image_name = []
    image_name.extend(os.listdir(os.path.join(input_path, sub_file)))
    image_name.sort()
    
    for i_image in xrange(len(image_name)):

        # determin the class type
        cloth_type = image_name[i_image]
        cloth_type = cloth_type.split('_')[-2].split('.')[0]

        this_type = -1
        
        for sub_type in subcategories:
            if cloth_type in subcategories[sub_type]:
                this_type = sub_type
                break
        
        if this_type == -1:
            # TODO, record the image
            error_parsing_file.write(os.path.join( \
                os.path.basename(input_path), sub_file, 
                image_name[i_image]) + '\n')
            continue
        result_file.write(os.path.join(os.path.basename(input_path), sub_file, 
                image_name[i_image]) + ' ' + str(this_type) + ' ')
        timer = Timer()  # time the detection
        timer.tic()
        
        image_file = os.path.join(input_path, sub_file, image_name[i_image])
        if not image_file.endswith('.jpg'):
            continue
        
        if not os.path.exists(image_file):
            continue
        im = cv2.imread(image_file)
        if im is None:
            continue

        # generate the proposal files        
        os.system(
            '/media/DataDisk/twwang/fast-rcnn/rcnn_test/proposals_for_python.sh' \
            + ' ' + os.path.join(input_path, sub_file, image_name[i_image]))
            
        data = open('/home/twwang/temp_proposal', "rb").read()
        number_proposals = struct.unpack("i", data[0:4])[0]
        number_edge = struct.unpack("i", data[4:8])[0]
        assert number_edge == 4, 'The size is not matched!\n' + \
            'Note that the first two variables are the number of proposals\n' + \
            ' and number of coordinates in a box, which is 4 by default\n'

        number_proposals = min(cfg.NUM_PPS, number_proposals)
        obj_proposals = np.asarray(struct.unpack(
            str(number_proposals * 4) + 'f',
            data[8: 8 + 16 * number_proposals])).reshape(number_proposals, 4)
            
        # it is a better idea to set a TEST_MULTI_LABEL
        if cfg.MULTI_LABEL:
            scores, boxes, multi_label = im_detect(net, im, obj_proposals)
        else:
            scores, boxes = im_detect(net, im, obj_proposals)

        # for the class, we go for a nms
        cls_ind = this_type
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
            cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)

        # keep the needed
        dets = dets[keep, :]

        # get the sorted results
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds, :]

        # get the data together
        cls_line = np.ones((dets.shape[0], 1)) * cls_ind

        results = dets.astype(np.float32)
        results_cls = cls_line.astype(np.int32)          

        # sort the results
        scores = results[:, 4]
        order = scores.argsort()[::-1]
        
        results = results[order, :]
        results_cls = results_cls[order, :]
        
        # write the result into the result files
        if results.shape[0] >= 1:        
            result_file.write(str(results[0, 0]) + ' ' + str(results[0, 1]) + \
                ' ' + str(results[0, 2]) + ' ' + str(results[0, 3]) + '\n')
        else:
            result_file.write('-1 -1 -1 -1\n')
        # plot the image if necessary                
        for i in xrange(0, results.shape[0]):
            if results[i, 4] < PLOT_CONF_THRESH:
                continue
            if results_cls[i, 0] == 1:
                cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (0,255,0), 3)
                cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                            ',P: ' + str(results[i, 4]),
                            (results[i, 0], results[i, 1]),
                            font, 1, (0,255,0), 2, 255)
            else:
                if results_cls[i, 0] == 2:
                    cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (255,0,0), 3)
                    cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                        ',P: ' + str(results[i, 4]),
                        (results[i, 0], results[i, 1]),
                        font, 1, (255,0,0), 2, 255)
                else:
                    cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (0,0,255), 3)
                    cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                        ',P: ' + str(results[i, 4]),
                        (results[i, 0], results[i, 1]),
                        font, 1, (0,0,255), 2, 255)
                        
        cv2.imwrite(os.path.join(output_path, 'images', 
            sub_file, image_name[i_image]), im)
            
        timer.toc()
        print("The running time is {} on the {} th image"\
            .format(timer.total_time, i_image))
    result_file.close()
    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='test a Fast R-CNN network')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
        help='Use CPU mode (overrides --gpu)',
        action='store_true')
    parser.add_argument('--model', dest='Caffenet', help='',
        default='0')
    parser.add_argument('--prototxt', dest='prototxt', help='',
        default='0')
    parser.add_argument('--plotImage', dest='plot', help='',
        default='True')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.Caffenet == 0 or args.prototxt == 0:
        print("Specify the network path!")
        sys.exit()
    
    # loading the model and initialize it
    prototxt = args.prototxt
    caffemodel = args.Caffenet

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                            'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
            
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('\n\nLoaded network {:s}'.format(caffemodel))
 
    # the output will be save to another directory, the result is in two 
    # different subdirectories. One is the `txt`, one is the `images`
    input_path = '/media/DataDisk/twwang/GoogleImages_topcloth'
    output_path = input_path + '_result'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    if not os.path.exists(os.path.join(output_path, 'images')):
        os.mkdir(os.path.join(output_path, 'images'))

    # the error parsing file
    error_parsing_file = open(os.path.join(output_path, 'error_file' + '_' + \
        str(datetime.datetime.now().time()) + '.txt'), 'w')
        
    for sub_file in os.listdir(input_path):
        google_top(net, input_path, sub_file, output_path)
    
    error_parsing_file.close()