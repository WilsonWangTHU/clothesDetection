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
import PIL

num_class = 3
root_dir = '/media/Elements/twwang/fast-rcnn/data/clothesDataset/test'

data_dir = '/media/Elements/twwang/fast-rcnn/data/CCP'

top_number = 5

CONF_THRESH = 0.4
NMS_THRESH = 0.3
font = cv2.FONT_HERSHEY_SIMPLEX
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

def fowever21test(net):

    # reading the file names in the mapping files
    image_dir = os.path.join(data_dir, 'photos')
    proposal_dir = os.path.join(data_dir, 'proposals')

    # get the image name
    image_name = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # get the proposals
    # Load image one by one
    for i_image in xrange(0, len(image_name)):
        image_file = os.path.join(data_dir, \
                'photos', image_name[i_image])
        proposal_file = os.path.join(data_dir, \
                'proposals', 'proposals' + image_name[i_image])
        
        data = open(proposal_file, "r")
        number_proposals = int(data.readline())
        number_edge = int(data.readline())
        number_proposals = min(cfg.NUM_PPS, number_proposals)
        proposal_data = np.zeros((number_proposals, 4), dtype=np.float32)
        for i in xrange(number_proposals):
            proposal_data[i, :] = np.float32(data.readline().strip().split())
            
        im = cv2.imread(image_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im, proposal_data)
        timer.toc()
        print("The running time is {} on the {} th image".format(timer.total_time, i_image))
        
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
        for i in xrange(0, results.shape[0]):
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
        cv2.imwrite(os.path.join(data_dir, \
                'results_resized', image_name[i_image]), im)
        #if i_image == 130:
         #   raw_input()
        
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
    fowever21test(net)
