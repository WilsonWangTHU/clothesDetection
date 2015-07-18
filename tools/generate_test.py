# -------------------------------------------------------
#
#   it is a python port of network test, generate the 
#   results of the three class: Upper body(1), Lower 
#   body (2), and the whole body(3)
#
#   Written by Tingwu Wang, 17.7.2015
#
# -------------------------------------------------------

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import xml.dom.minidom as minidom
import struct


num_category = 26
num_class = 3
root_dir = '/media/Elements/twwang/fast-rcnn/data/clothesDataset/test'

top_number = 5

accuracy_perCls_top1 = np.zeros((num_category), dtype=np.float)
accuracy_perCls_top5 = np.zeros((num_category), dtype=np.float)

accuracy_perCls_top1_box = np.zeros((num_category), dtype=np.float)
accuracy_perCls_top5_box = np.zeros((num_category), dtype=np.float)

pred_preCls = np.zeros((26, 26), dtype=np.float)
CONF_THRESH = 0.4
NMS_THRESH = 0.3

CLASSES = ('__background__', 'Upper', 'Lower', 'Whole')


def readcloth_name(image_set_file):
    assert os.path.exists(image_set_file), \
            'index txt does not exist: {}{}'.format(image_set_file)
    image_name = []

    with open(image_set_file) as f:
        for x in f.readlines():
            # when using the three class, it is a different label way
            y = x.strip()
            if y.find("\"") == -1:  # error reading the files!
                print("Reading a wrong GUID files at {}!".format( \
                        image_set_file))
                sys.exit()
            else:
                image_name.append(y[y.find("/") + 1: y.find("\"")])
    return image_name

def readcloth_proposals(image_name, category):
    proposals  = []

    # get the proposal files image by image 
    for i in xrange(0, len(image_name)):
        proposal_files = os.path.join(root_dir, str(category), \
                "proposals", image_name[i])
        if not os.path.isfile(caffemodel):
            raise IOError()

        # get the number of proposals
        if category == 1:
            # i made a mistake, the 1st cat is in text file
            data = open(proposal_files, "r")
            number_proposals = int(data.readline())
            number_edge = int(data.readline())
            number_proposals = min(cfg.NUM_PPS, number_proposals)
            raw_data = np.zeros((number_proposals, 4), dtype=np.float32)
            for i in xrange(number_proposals):
                raw_data[i, :] = np.float32(data.readline().strip().split())
        else:
            data = open(proposal_files, "rb").read()
            number_proposals = struct.unpack("i", data[0:4])[0]
            number_edge = struct.unpack("i", data[4:8])[0]
            number_proposals = min(cfg.NUM_PPS, number_proposals)
            raw_data = np.asarray(struct.unpack(str(number_proposals * 4) \
                + 'f',data[8: 8 + 16 * number_proposals])).\
                reshape(number_proposals, 4)
                
        proposals.append(raw_data[:, :])

    return proposals

def category_test(category, net):

    # reading the file names in the mapping files
    image_set_file = os.path.join(root_dir, str(category), 'newGUIDMapping.txt')
    image_name = readcloth_name(image_set_file)

    # get the proposals
    image_proposal = readcloth_proposals(image_name, category)

    # get the writter
    floatwritter = open(os.path.join(root_dir, str(category),
                                     'result', 'floatResults'), 'w')
    intwritter = open(os.path.join(root_dir, str(category),
                                   'result', 'intResults'), 'w')
    floatwritter.write(str(len(image_name)) + '\n')
    floatwritter.write(str(5) + '\n')    
    intwritter.write(str(len(image_name)) + '\n')
    intwritter.write(str(1) + '\n')

    # Load image one by one
    for i_image in xrange(0, len(image_name)):
        image_file = os.path.join(root_dir, \
                str(category), 'images', image_name[i_image])
        im = cv2.imread(image_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im, image_proposal[i_image])
        timer.toc()
        print("The running time is {} on the {} th image of categary {}".format(timer.total_time, i_image, category))

        # for each class, we go for a nms
        for cls in xrange(1, num_class):
            cls_ind = cls
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            # get the sorted results
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = dets[inds, :]
            # get the data together
            cls_line = np.ones((dets.shape[0], 1)) * cls

            if cls == 1:
                results = dets.astype(np.float32)
                results_cls = cls_line.astype(np.int32)
            else:
                results = np.vstack((results, dets)).astype(np.float32)
                results_cls = np.vstack((results_cls, cls_line)).astype(np.int32)

        # sort the results
        scores = results[:, 4]
        order = scores.argsort()[::-1]
        
        results = results[order, :]
        results_cls = results_cls[order, :]
        #if i_image == 130:
         #   raw_input()

        for i in xrange(0, top_number):
            if i >= results.shape[0]:
                floatwritter.write(str(-1) + ' ')
                floatwritter.write(str(-1) + ' ')
                floatwritter.write(str(-1) + ' ')
                floatwritter.write(str(-1) + ' ')
                floatwritter.write(str(-1) + ' ')
                intwritter.write(str(-1) + ' ')
            else:
                floatwritter.write(str(results[i, 0]) + ' ')
                floatwritter.write(str(results[i, 1]) + ' ')
                floatwritter.write(str(results[i, 2]) + ' ')
                floatwritter.write(str(results[i, 3]) + ' ')
                floatwritter.write(str(results[i, 4]) + ' ')
                intwritter.write(str(results_cls[i, 0]) + ' ')
            intwritter.write('\n')
            floatwritter.write('\n')

    floatwritter.close();
    intwritter.close();
    return

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='test a Fast R-CNN network')

    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
        help='Use CPU mode (overrides --gpu)',
        action='store_true')
    parser.add_argument('--model', dest='Caffenet', help='',
        default='0')
    parser.add_argument('--prototxt', dest='prototxt', help='',
        default='0')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

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
 
    # test for each category
    for iNum in xrange(1, num_category+1):
        category_test(iNum, net)
