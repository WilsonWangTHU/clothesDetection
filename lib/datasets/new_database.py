#coding=utf-8
# -----------------------------------------------------------------------------
# Fast rcnn clothes detector
# @Brief: In this function, the raw dataset is written for further uses. The 
# function is not well organized due to its origianal architecture. We may need
# to change the lay out in th future.
# 
# Written by Tingwu Wang, from 01, 05, 2015, i.e. during the internship in
# SenseTime, Beijing
#
# The original fast-rcnn detector is under the 
#   copyright 2015 Microsoft, licensed under The MIT License 
#   Written by Ross Girshick
#
# Update: 01/08/2015, the multilabel attributive is added
# Update: 08/08/2015, a better way to pick the foreground is introduced
# Update: 14/08/2015, I try to add the CCP and CFD dataset into the dataset
# Update: 25/08/2015,
# -----------------------------------------------------------------------------

from math import floor
import datasets
import datasets.new_database
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import PIL
import struct
from fast_rcnn.config import cfg
import sys

# this is a debuger to make sure the third class is working
WHOLE_DEBUGER = {8 : 1, 11 : 2, 20: 3}
TYPE_MAPPER = {1 : [1,2,3,4,5,6,7,9,10,12,13,14,15,16,17,18,19], 
               2 : [21,22,23,24,25,26], 
               3 : [8, 11, 20]}

def twentysix2three(class_type):
    if cfg.SEP_DETECTOR:
        return 1
    if cfg.DEBUG_CLASS_WHOLE == True:
        return WHOLE_DEBUGER[class_type]
    if 1 <= class_type <= 7 or 9 <= class_type <= 10 or 12 <= class_type <= 19:
        out_number = 1  # 17 class 1 type
    else:
        if class_type == 8 or class_type == 11 or class_type == 20:
            out_number = 3  # three class 3 types
        else:
            out_number = 2  # 7 class 2 type
    return out_number

class new_database(datasets.imdb):

    def __init__(self, db_name, stage, devkit_path=None):
        datasets.imdb.__init__(self, db_name)
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path

        # get the dataset name, it will be useful in the future use
        self.dataset_name = db_name

        # different data has different path. Note that the JD dataset is 
        # particular strange and may need some time to understand
        self._data_path = os.path.join(self._devkit_path, db_name)
        self._stage = stage

        # the class type variables
        self._type_classes = ('风衣', '毛呢大衣', '羊毛衫/羊绒衫',
                              '棉服/羽绒服',  '小西装/短外套',
                              '西服', '夹克', '旗袍', '皮衣', '皮草',
                              '婚纱', '衬衫', 'T恤', 'Polo衫', '开衫',
                              '马甲', '男女背心及吊带', '卫衣',
                              '雪纺衫', '连衣裙', '半身裙',
                              '打底裤', '休闲裤', '牛仔裤', '短裤',
                              '卫裤/运动裤'
                              )
        self._texture_classes = ('一致色', '横条纹', '纵条纹',
                                 '豹纹斑马纹', '格子',
                                 '圆点', '乱花', 'LOGO及印花图案', '其他'
                                 )
        self._neckband_classes = ('圆领', 'V领', '翻领',
                                  '立领', '高领', '围巾领',
                                  '一字领', '大翻领西装领',
                                  '连帽领', '其他'
                                  )
        self._sleeve_classes = ('短袖', '中袖', '长袖')
        self._ts_classes = ('__background__', '风衣', '毛呢大衣',
                              '羊毛衫/羊绒衫', '棉服/羽绒服',  '小西装/短外套',
                              '西服', '夹克', '旗袍', '皮衣', '皮草',
                              '婚纱', '衬衫', 'T恤', 'Polo衫', '开衫',
                              '马甲', '男女背心及吊带', '卫衣',
                              '雪纺衫', '连衣裙', '半身裙',
                              '打底裤', '休闲裤', '牛仔裤', '短裤',
                              '卫裤/运动裤'
                              )

        # for the three class detection, 1 for upper, 2 for lower, and
        # three for clothes that covers the whole body
        if cfg.ThreeClass == True:
            self._classes = ('__background__', 'Upper', 'Lower', 'Whole')
            if cfg.DEBUG_CLASS_WHOLE == True:
                self._classes = ('__background__', '8', '11', '20', 'test')
        else:
            self._classes = self._ts_classes

        # the name of the label file extension
        self._label_ext = '.clothInfo'
        self._len_sleeve_cls = len(self._sleeve_classes)
        self._len_texture_cls = len(self._texture_classes)
        self._len_neckband_cls = len(self._neckband_classes)
        self._len_label = self._len_sleeve_cls + self._len_neckband_cls + \
                self._len_texture_cls

        # the index files to map the name of the class to the numbers
        self._class_to_ind = \
                dict(zip(self._classes, xrange(len(self._classes))))
        self._ts_class_to_ind = \
                dict(zip(self._ts_classes, xrange(len(self._ts_classes))))
        self._texture_to_label_ind = \
                dict(zip(self._texture_classes,
                    xrange(self._len_texture_cls)))
        self._neckband_to_label_ind = \
                dict(zip(self._neckband_classes,
                    xrange(self._len_texture_cls,
                        self._len_texture_cls + self._len_neckband_cls)
                    ))
        self._sleeve_to_label_ind = \
                dict(zip(self._sleeve_classes,
                    xrange(self._len_texture_cls + self._len_neckband_cls,
                        self._len_label)   
                    ))

        # the name of the image, the type of the image, and the name of 
        # its label file. The index is going to be extended into the image
        # path. The self._image_type and the self._image_twentysix_type
        # is useful for the path in the JD data set
        # The self._image_label is going to be extended into the label
        # file index
        self._image_index, self._image_type, self._image_label, \
            self._image_twentysix_type, = self._load_dataset_index_and_path()

        # Default to roidb handler, it is no more useful now
        self._roidb_handler = self.roidb_loading_interface

        # specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, i):
        """
        Construct an image path from the image's "index" identifier.
        """
        if self.dataset_name == 'clothesDataset':
            image_path = os.path.join(self._data_path, self._stage,
                str(self._image_twentysix_type[i]), str(self._image_index[i]))
        else:
            if self.dataset_name in ['CFD', 'CCP', 'Fashionista']:
                image_path = os.path.join(self._devkit_path, self.dataset_name,
                    'images', self._image_index[i])
            else:
                assert 1 == 2, \
                        "Error! Unknow dataset!"

        assert os.path.exists(image_path), \
            'image file does not exist: {}{}'.format(image_path, i)
        return image_path

    def _load_dataset_index_and_path(self):
        """
        Load the indexes listed in this dataset's image set file.  there 
        are more than 26 sub files in the system use the type_classes to
        read all the image!
        # Example path to JD image set file: 
        # fast-rcnn/data/clothesDataset/train/1/images/xxxx.jpg
        # fast-rcnn/data/CFD/images/xxxx.jpg
        # fast-rcnn/data/CCP/images/xxxx.jpg
        """

        image_index = []
        image_type = []
        image_label = []
        image_twentysix_type = []

        if self.dataset_name == 'clothesDataset':
            # load the JD clothes dataset. Note that the dataset is divided in
            # 25 sub classes, so it is necessary to 
            class_list = list(xrange(1, len(self._type_classes) + 1))
            if cfg.BALANCED == True:
                # append more class three type to balance the datasets
                # type 8, 11, 20 is the class three, add then to 12 types in all
                append_list_8 = list(8 * np.ones(cfg.BALANCED_COF, dtype=np.int32))
                append_list_11 = list(11 * np.ones(cfg.BALANCED_COF, dtype=np.int32))
                append_list_20 = list(20 * np.ones(cfg.BALANCED_COF, dtype=np.int32))
                class_list.extend(append_list_8)
                class_list.extend(append_list_11)
                class_list.extend(append_list_20)
            if cfg.DEBUG_CLASS_WHOLE == True:
                class_list = [8, 11, 20]
            
            if cfg.SEP_DETECTOR:
                class_list = TYPE_MAPPER[cfg.SEP_DETECTOR]
                
            for class_type in class_list:
                # the twenty six type is useful when loading the annotations
                image_set_file = os.path.join(self._data_path, self._stage, str(class_type),
                                              'newGUIDMapping.txt')
                assert os.path.exists(image_set_file), \
                    'index txt does not exist: {}'.format(image_set_file)
                with open(image_set_file) as f:
                    for x in f.readlines():
                        # when using the three class, it is a different label way
                        y = x.strip()
                        image_twentysix_type.append(class_type)
    
                        if cfg.ThreeClass == False:
                            image_type.append(class_type)
                        else:
                            image_type.append(twentysix2three(class_type))
                        if y.find('.jpg') == -1:  # it is not a jpg file
                            image_label.append(y[y.find('.png') + 1 + 4:])
                            image_index.append(y[y.find('\\') + 1: y.find('.png')] + '.png')
                        else:
                            image_label.append(y[y.find('.jpg') + 1 + 4:])
                            image_index.append(y[y.find('\\') + 1: y.find('.jpg')] + '.jpg')
                            
        else:
            if self.dataset_name in ['CCP', 'CFD', 'Fashionista']:
                image_index.extend(os.listdir(os.path.join(
                    self._devkit_path, self.dataset_name, 'images')))
                image_label = image_index
                image_twentysix_type = image_index
                image_type = image_index
            else:
                assert 1 == 2, "Unknown dataset {}!".format(self.dataset_name)

        return image_index, image_type, image_label, image_twentysix_type

    def _get_default_path(self):
        """
        Return the default path of closes.
        """
        return os.path.join(datasets.ROOT_DIR, 'data')
        
    def roidb_loading_interface(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if not cfg.SEP_DETECTOR:
            cache_file = os.path.join(self.cache_path, \
                self.name + '_3CL=' + str(cfg.ThreeClass) + \
                '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
                '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
                '_BLC=' + str(cfg.BALANCED) + \
                '_COF=' + str(cfg.BALANCED_COF) + \
                '_TT1000=' + str(cfg.TESTTYPE1000) + \
                '_roidb.pkl')
        else:
            cache_file = os.path.join(self.cache_path, \
                self.name + '_3CL=' + str(cfg.ThreeClass) + \
                '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
                '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
                '_BLC=' + str(cfg.BALANCED) + \
                '_COF=' + str(cfg.BALANCED_COF) + \
                '_TT1000=' + str(cfg.TESTTYPE1000) + \
                '_SEP_DETECTOR' + str(cfg.SEP_DETECTOR_NUM) + \
                '_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        # it is changed, as we are not loading the selective search
        # any more, we used the pre-computed Edge box, it's faster
        gt_roidb = self._load_groundtruth_roidb()
        ss_roidb = self._load_proposal_roidb(gt_roidb)

        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

        print("The ROIDB are all loaded and merged with the gt!")

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_groundtruth_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if not cfg.SEP_DETECTOR:
            cache_file = os.path.join(self.cache_path, \
                self.name + '_3CL=' + str(cfg.ThreeClass) + \
                '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
                '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
                '_BLC=' + str(cfg.BALANCED) + \
                '_COF=' + str(cfg.BALANCED_COF) + \
                '_TT1000=' + str(cfg.TESTTYPE1000) + \
                '_gt_roidb.pkl')
        else:
            cache_file = os.path.join(self.cache_path, \
                self.name + '_3CL=' + str(cfg.ThreeClass) + \
                '_MULTI_LABEL=' + str(cfg.MULTI_LABEL) + \
                '_SOFTMAX=' + str(cfg.MULTI_LABEL_SOFTMAX) + \
                '_BLC=' + str(cfg.BALANCED) + \
                '_COF=' + str(cfg.BALANCED_COF) + \
                '_TT1000=' + str(cfg.TESTTYPE1000) + \
                '_SEP_DETECTOR' + str(cfg.SEP_DETECTOR_NUM) + \
                '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self.dataset_name == 'clothesDataset':
            gt_roidb = [self._load_JD_dataset_annotation(i)
                    for i in xrange(len(self.image_index))]
        else:
            assert self.dataset_name in ['CCP', 'CFD', 'Fashionista'], \
                    '{} is an UNKNOWN dataset!'.format(self.dataset_name)
            gt_roidb = [self._load_CFDCCP_dataset_annotation(i)
                    for i in xrange(len(self.image_index))]
        print 'Finish loading annotations'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_proposal_roidb(self, gt_roidb):
        box_list = []
        # read data from each sub file, the _image_twentysix_type is 
        # important to clearify the sub directory
        for i in xrange(len(self._image_index)):
            # the first 7 char of index is the 'images\' take it out
            if self.dataset_name == 'clothesDataset':
                filename = os.path.join(self._data_path, self._stage,
                    str(self._image_twentysix_type[i]),
                    'proposals', self._image_index[i][7:])
            else:
                assert self.dataset_name in ['CCP', 'CFD', 'Fashionista'], \
                    'Unknow dataset {}'.format(self.dataset_name)
                filename = os.path.join(self._devkit_path, self.dataset_name, 
                    'proposals', self._image_index[i])
            assert os.path.exists(filename), \
                'Proposal box data not found at: {}'.format(filename)
            
            if self.dataset_name in ['CCP', 'CFD', 'Fashionista']:
                data = open(filename, "r")
                number_proposals = int(data.readline())
                number_edge = int(data.readline())
                number_proposals = min(cfg.NUM_PPS, number_proposals)
                raw_data = np.zeros((number_proposals, 4), dtype=np.float32)
                for i in xrange(number_proposals):
                    raw_data[i, :] = np.float32(data.readline().strip().split())
            else:
                assert self.dataset_name == 'clothesDataset', \
                    'Unknown dataset {}'.format(self.dataset_name)
                data = open(filename, "rb").read()
                number_proposals = struct.unpack("i", data[0:4])[0]
                number_edge = struct.unpack("i", data[4:8])[0]
            
                number_proposals = min(cfg.NUM_PPS, number_proposals)
                raw_data = np.asarray(struct.unpack(str(number_proposals * 4) \
                        + 'f', data[8: 8 + 16 * number_proposals])).\
                        reshape(number_proposals, 4)

            box_list.append(raw_data[:, :])
        
        print("The proposals are all loaded")

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_CFDCCP_dataset_annotation(self, i):
        if i % 100 == 0: 
            print("Now loading annotations of the {} th image".format(i))
        mat_file_name = self._image_label[i][:self._image_label[i].find('.')] \
                + '.mat'
        filename = os.path.join(self._devkit_path, self.dataset_name, 
                'bounding_box', mat_file_name)
        raw_data = sio.loadmat(filename)['coordinates']
        assert raw_data.shape[0] == 4 and raw_data.shape[1] == 5, \
                "The proposal mat file seems to be broke at {}"\
                .format(filename)

        # get the number of groundtruth objects in the data
        num_objs = np.where(raw_data[:,2] < raw_data[:,4])[0].shape[0]

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)


        # they are just for compatibility purpose
        assert cfg.ATTR_CHOICE == False and cfg.MULTI_LABEL == False \
            and cfg.MULTI_LABEL_SOFTMAX == False, \
            "Only the JD clothesDataset support MULTI_LABEL and ATTR_CHOICE"

        obj_counter = 0
        for ind in np.where(raw_data[:,2] < raw_data[:,4])[0]:
            # load each gt class class and its coordinates
            boxes[obj_counter, :] = raw_data[ind, 1:]
            gt_classes[obj_counter] = raw_data[ind, 0]
            overlaps[obj_counter, raw_data[ind, 0]] = 1.0

            obj_counter = obj_counter + 1
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _load_JD_dataset_annotation(self, i):
        # Load image and bounding boxes info from the JD dataset XML file
        if i % 100 == 0: 
            print("Now loading annotations of the {} th image".format(i))
        filename = os.path.join(self._data_path, self._stage,
                str(self._image_twentysix_type[i]),
                'Label', self._image_label[i] + self._label_ext)
        # read the width and height of the image make sure the annotation
        # is within the picture
        im = PIL.Image.open(self.image_path_at(i))
        widths = im.size[0]
        height = im.size[1]

        with open(filename) as f:
            data = minidom.parseString(f.read())

        type_objs = data.getElementsByTagName('clothClass')

        # nType always = 1 here, which is the case of our dataset
        num_objs = 1

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        if cfg.ATTR_CHOICE:
            sleeve_coordinate1 = -1 * np.ones((num_objs, 4), dtype=np.uint16)
            sleeve_coordinate2 = -1 * np.ones((num_objs, 4), dtype=np.uint16)
            neckband_coordinate = -1 * np.ones((num_objs, 4), dtype=np.uint16)
        if cfg.MULTI_LABEL == True:
            if not cfg.MULTI_LABEL_SOFTMAX:
                multi_label = \
                        -1 * np.ones((num_objs, self._len_label), dtype=np.int32)
            else:
                multi_label = \
                        np.zeros((num_objs, self._len_label), dtype=np.int32)


        # Load object bounding boxes into a data frame. first we load the
        # type class and the texture class
        for ix, obj in enumerate(type_objs):
            # add type class and texture class simultaneously
            # Make pixel indexes 0-based
            location = obj.getElementsByTagName('Location')[0]
            
            validation = location.getAttributeNode('SourceQuality').childNodes[0].data
            if validation != u'Valid':
                # this box is useless, error!
                sys.exit("Error! An invalid tags find at {}".format(self.image_path_at(i)))
            else:
                x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
                y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
                x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
                y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))
    
                # make sure the coordinates are within the pic
                x1 = min(x1, widths - 1)
                x2 = min(x2, widths - 1)
                y1 = min(y1, height - 1)
                y2 = min(y2, height - 1)
    
                # the class number of the proposals
                cls_str = obj.getAttributeNode('type').childNodes[0].data
                cls_str = cls_str.encode('utf-8')  # change the unicode into str

                # 3 classes r 26 classes?
                if cfg.ThreeClass == False:
                    cls = self._ts_class_to_ind[cls_str]
                else:
                    cls = twentysix2three(self._ts_class_to_ind[cls_str])

            # take out the possible label error of wrong orders
            if x1 > x2:
                temp_value = x1
                x1 = x2
                x2 = temp_value
            if y1 > y2:
                temp_value = y1
                y1 = y2
                y2 = temp_value

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if cfg.MULTI_LABEL == False:
            return {'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': False}

        # now get the label information if we use the multi-label
        type_objs = data.getElementsByTagName('clothTexture')
        if len(type_objs) != 0:
            label_type = type_objs[0]
            label_cls = \
                    label_type.getAttributeNode('type').childNodes[0].data
            label_cls = label_cls.encode('utf-8')
            if self._texture_to_label_ind.has_key(label_cls):
                label_cls = self._texture_to_label_ind[label_cls]
                multi_label[0, label_cls] = 1;
        
        # we dont consider one cloth with two different kinds of 
        type_objs = data.getElementsByTagName('clothNeckband')
        if len(type_objs) != 0:
            label_type = type_objs[0]
            label_cls = \
                    label_type.getAttributeNode('type').childNodes[0].data
            label_cls = label_cls.encode('utf-8')
            if self._neckband_to_label_ind.has_key(label_cls):
                label_cls = self._neckband_to_label_ind[label_cls]
                multi_label[0, label_cls] = 1;
        if cfg.ATTR_CHOICE:
            for i_neckband in xrange(len(type_objs)):
                location = type_objs[i_neckband].getElementsByTagName('Location')[0]
                validation = location.getAttributeNode('SourceQuality').childNodes[0].data
                if validation != u'Valid':
                    # this box is useless, continue!
                    continue
                else:
                    x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
                    y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
                    x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
                    y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))
        
                    # make sure the coordinates are within the pic
                    x1 = min(x1, widths - 1)
                    x2 = min(x2, widths - 1)
                    y1 = min(y1, height - 1)
                    y2 = min(y2, height - 1)
                    neckband_coordinate[0, :] = \
                        [min(x1, x2), min(y1, y2), 
                        max(x1, x2), max(y1, y2)]

        # the sleeve type
        type_objs = data.getElementsByTagName('clothSleeve')
        if len(type_objs) != 0:
            label_type = type_objs[0]
            label_cls = \
                    label_type.getAttributeNode('type').childNodes[0].data
            label_cls = label_cls.encode('utf-8')
            if self._sleeve_to_label_ind.has_key(label_cls):
                label_cls = self._sleeve_to_label_ind[label_cls]
                multi_label[0, label_cls] = 1;

        # get the attributive location if necessary
        if cfg.ATTR_CHOICE:
            for i_sleeve in xrange(len(type_objs)):
                location = type_objs[i_sleeve].getElementsByTagName('Location')[0]
                validation = location.getAttributeNode('SourceQuality').childNodes[0].data
                if validation != u'Valid':
                    # this box is useless, error!
                    continue
                else:
                    x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
                    y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
                    x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
                    y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))
        
                    # make sure the coordinates are within the pic
                    x1 = min(x1, widths - 1)
                    x2 = min(x2, widths - 1)
                    y1 = min(y1, height - 1)
                    y2 = min(y2, height - 1)
                    if i_sleeve == 1:
                        sleeve_coordinate1[0, :] = \
                                [min(x1, x2), min(y1, y2), 
                                        max(x1, x2), max(y1, y2)]
                    else:
                        sleeve_coordinate2[0, :] = \
                                [min(x1, x2), min(y1, y2), 
                                        max(x1, x2), max(y1, y2)]

        if not cfg.ATTR_CHOICE:
            return {'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': False,
                    'multi_label': multi_label}

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'sleeve_coordinate1': sleeve_coordinate1,
                'sleeve_coordinate2': sleeve_coordinate2,
                'neckband_coordinate': neckband_coordinate,
                'flipped': False,
                'multi_label': multi_label}

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
