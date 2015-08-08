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
import caffe, os, sys, cv2
import argparse
import struct

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

font = cv2.FONT_HERSHEY_SIMPLEX

num_category = 26
num_class = 3

Jingdong_root_dir = '/media/DataDisk/twwang/fast-rcnn/data/clothesDataset/test'
Jingdong_output_dir = '/media/DataDisk/twwang/fast-rcnn/data/results/Jingdong_bg'

forever21_data_dir = '/media/DataDisk/twwang/fast-rcnn/data/CCP'
forever21_output_dir = '/media/DataDisk/twwang/fast-rcnn/data/results/forever21'

top_number = 10

accuracy_perCls_top1 = np.zeros((num_category), dtype=np.float)
accuracy_perCls_top5 = np.zeros((num_category), dtype=np.float)

accuracy_perCls_top1_box = np.zeros((num_category), dtype=np.float)
accuracy_perCls_top5_box = np.zeros((num_category), dtype=np.float)

pred_preCls = np.zeros((26, 26), dtype=np.float)

CONF_THRESH = 0.1
PLOT_CONF_THRESH = 0.4
PLOT_MULTI_LABEL_THRESH = 0.4
NMS_THRESH = 0.3

CLASSES = ('__background__', 'Upper', 'Lower', 'Whole')

def get_attributive_string(attributive_vector):
    
    if np.max(attributive_vector[\
            0 : len(texture_to_label_ind)]) > PLOT_MULTI_LABEL_THRESH:
        attributive_string_texture = \
                texture_to_label_ind[np.argmax(attributive_vector[\
                0 : len(texture_to_label_ind)])]
    else:
        attributive_string_texture = '-1 '
    if np.max(attributive_vector[\
            len(texture_to_label_ind) : \
            len(neckband_to_label_ind) + len(texture_to_label_ind)]) \
            > PLOT_MULTI_LABEL_THRESH:
        attributive_string_neckband = \
                neckband_to_label_ind[np.argmax(attributive_vector[ \
                len(texture_to_label_ind) : \
                len(neckband_to_label_ind) + len(texture_to_label_ind)])]
    else:
        attributive_string_neckband = '-1 '
    if np.max(attributive_vector[\
            len(neckband_to_label_ind) + len(texture_to_label_ind) \
            : len(texture_to_label_ind) + len(neckband_to_label_ind) + \
            len(sleeve_to_label_ind)]) \
            > PLOT_MULTI_LABEL_THRESH:
        attributive_string_sleeve = \
                sleeve_to_label_ind[np.argmax(attributive_vector[ \
                len(texture_to_label_ind) + len(neckband_to_label_ind) \
                : len(texture_to_label_ind) + len(neckband_to_label_ind) + \
                len(sleeve_to_label_ind)])]
    else:
        attributive_string_sleeve = '-1 '
    return attributive_string_texture, attributive_string_neckband, \
        attributive_string_sleeve

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
        proposal_files = os.path.join(Jingdong_root_dir, str(category), \
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

def fowever21test(net, args):

    # reading the file names in the mapping files
    image_dir = os.path.join(forever21_data_dir, 'photos')
    #proposal_dir = os.path.join(forever21_data_dir, 'proposals')

    # get the image name
    image_name = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # get the proposals
    # Load image one by one
    for i_image in xrange(0, len(image_name)):
        image_file = os.path.join(forever21_data_dir, \
                'photos', image_name[i_image])
        proposal_file = os.path.join(forever21_data_dir, \
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
        print("The running time is {} on the {} th image".\
            format(timer.total_time, i_image))
        
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
        
        # now save the results, note that fowever 21's results are stored 1 
        # file for 1 image, and the Jingdong results are stored 1 category for
        # 1 image
        floatwritter = open(os.path.join(
            forever21_output_dir, image_name[i_image] + '_floatResults'), 'w')
        intwritter = open(os.path.join(
            forever21_output_dir, image_name[i_image] +'_intResults'), 'w')

        # write the dimension of the data, it will be be compatible with the 
        # dbfeature tool used by the SenseTime
        floatwritter.write(str(top_number) + '\n')
        floatwritter.write(str(5) + '\n')
        intwritter.write(str(top_number) + '\n')
        intwritter.write(str(1) + '\n')

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
        
        # whether to plot the result demo image?
        if args.plot == False:
            continue
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
                        
        cv2.imwrite(os.path.join(forever21_output_dir, \
                'images', image_name[i_image]), im)
        
    return
    
    
def category_test(category, net, args):

    # reading the file names in the mapping files
    image_set_file = os.path.join(Jingdong_root_dir, str(category),
                                  'newGUIDMapping.txt')
    image_name = readcloth_name(image_set_file)

    # get the proposals
    image_proposal = readcloth_proposals(image_name, category)

    # get the writter
    floatwritter = open(os.path.join(Jingdong_output_dir, 
                                     str(category) + 'floatResults'), 'w')
    intwritter = open(os.path.join(Jingdong_output_dir, 
                                   str(category) + 'intResults'), 'w')

    # if not testing the multi_label, the float results only record 
    # the top 10 [x1, x2, y1, y2, prob].
    # When testing the multi_label, the float results is then top 10
    # [x1, x2, y1, y2, prob, attri(1:23) ]
    floatwritter.write(str(top_number * len(image_name)) + '\n')
    if not cfg.MULTI_LABEL:
        floatwritter.write(str(5) + '\n')
    else:
        floatwritter.write(str(5 + 22) + '\n')
    intwritter.write(str(top_number * len(image_name)) + '\n')
    intwritter.write(str(1) + '\n')

    # Load image one by one
    for i_image in xrange(0, len(image_name)):
        image_file = os.path.join(Jingdong_root_dir, \
                str(category), 'images', image_name[i_image])
        im = cv2.imread(image_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        # it is a better idea to set a TEST_MULTI_LABEL
        if cfg.MULTI_LABEL:
            scores, boxes, multi_label = \
                    im_detect(net, im, image_proposal[i_image])
        else:
            scores, boxes = \
                    im_detect(net, im, image_proposal[i_image])

        timer.toc()
        print("The running time is {} on the {} th image of categary {}".format(timer.total_time, i_image, category))

        # for each class, we go for a nms
        for cls in xrange(1, num_class + 1):
            cls_ind = cls
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)

            # keep the needed
            dets = dets[keep, :]
            if cfg.MULTI_LABEL:
                multi_label_this = multi_label[keep, :]

            # get the sorted results
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = dets[inds, :]
            if cfg.MULTI_LABEL:
                multi_label_this = multi_label_this[inds, :]

            # get the data together
            cls_line = np.ones((dets.shape[0], 1)) * cls

            if not cfg.MULTI_LABEL:
                if cls == 1:
                    results = dets.astype(np.float32)
                    results_cls = cls_line.astype(np.int32)
                else:
                    results = np.vstack((results, dets)).astype(np.float32)
                    results_cls = \
                            np.vstack((results_cls, cls_line)).astype(np.int32)
            else:
                if cls == 1:
                    results = \
                            np.hstack((dets, multi_label_this)).astype(np.float32)
                    results_cls = cls_line.astype(np.int32)
                else:
                    this_result = \
                            np.hstack((dets, multi_label_this)).astype(np.float32)
                    results = np.vstack((results, this_result)).astype(np.float32)
                    results_cls = \
                            np.vstack((results_cls, cls_line)).astype(np.int32)


        # sort the results
        scores = results[:, 4]
        order = scores.argsort()[::-1]
        
        results = results[order, :]
        results_cls = results_cls[order, :]

        if not cfg.MULTI_LABEL:
            for i in xrange(0, top_number):
                if i >= results.shape[0]:
                    for j in xrange(0, 5):
                        floatwritter.write(str(-1) + ' ')
                    intwritter.write(str(-1) + ' ')
                else:
                    for j in xrange(0, 5):
                       floatwritter.write(str(results[i, j]) + ' ')
                    intwritter.write(str(results_cls[i, 0]) + ' ')
                intwritter.write('\n')
                floatwritter.write('\n')
        else:
            for i in xrange(0, top_number):
                if i >= results.shape[0]:
                    for j in xrange(0, 5 + 22):
                        floatwritter.write(str(-1) + ' ')
                    intwritter.write(str(-1) + ' ')
                else:
                    for j in xrange(0, 5 + 22):
                       floatwritter.write(str(results[i, j]) + ' ')
                    intwritter.write(str(results_cls[i, 0]) + ' ')
                intwritter.write('\n')
                floatwritter.write('\n')
 
        # plot the image if necessary        
        if args.plot == False:
            continue
        for i in xrange(0, results.shape[0]):            
            if results[i, 4] < PLOT_CONF_THRESH:
                continue
            put_string1, put_string2, put_string3 = get_attributive_string(
                results[i, 5 : results.shape[1]])
                
            if results_cls[i, 0] == 1:
                cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (0,255,0), 3)
                cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                            ',P: ' + str(results[i, 4]),
                            (results[i, 0], results[i, 1]),
                            font, 0.8, (0,255,0), 1, 255)

            else:
                if results_cls[i, 0] == 2:
                    cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (255,0,0), 3)
                    cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                        ',P: ' + str(results[i, 4]),
                        (results[i, 0], results[i, 1]),
                        font, 0.8, (255,0,0), 1, 255)
                else:
                    cv2.rectangle(
                        im,(results[i, 0], results[i, 1]),
                        (results[i, 2], results[i, 3]),
                        (0,0,255), 3)
                    cv2.putText(im, "Cls: " + str(results_cls[i, 0]) + \
                        ',P: ' + str(results[i, 4]),
                        (results[i, 0], results[i, 1]),
                        font, 0.8, (0,0,255), 1, 255)
                        
            cv2.putText(im, put_string1,
                        (results[i, 0], int(results[i, 1] + 30)),
                        font, 0.8, (255,255,0), 1, 255)
            cv2.putText(im, put_string2,
                        (results[i, 0], int(results[i, 1] + 60)),
                        font, 0.8, (0,255,255), 1, 255)
            cv2.putText(im, put_string3,
                        (results[i, 0], int(results[i, 1] + 90)),
                        font, 0.8, (255,0,255), 1, 255)        
        cv2.imwrite(os.path.join(Jingdong_output_dir, \
                'images', str(category), image_name[i_image]), im)

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
    parser.add_argument('--testset', dest='dataset', help='',
        default='forever21')
    parser.add_argument('--plotImage', dest='plot', help='',
        default='True')
    parser.add_argument('--multi_label', dest='multi_label',
        help='', default='True')

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
 
    # change the output dir if necessary, and change the global 
    # flag
    if cfg.MULTI_LABEL:
        Jingdong_output_dir = Jingdong_output_dir + '_multi_label'
    if cfg.MULTI_LABEL_SOFTMAX:
        Jingdong_output_dir = Jingdong_output_dir + '_softmax'

    # test for each category
    if args.dataset == 'Jingdong':
        for iNum in xrange(1, num_category+1):
            category_test(iNum, net, args)
    if args.dataset == 'forever21':
        fowever21test(net, args)
        
