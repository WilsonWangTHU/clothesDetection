#coding=utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
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

''' in this file, we try to enable new database to be used'''

WHOLE_DEBUGER = {
    8 : 1,
    11 : 2,
    20: 3
}

def twentysix2three(class_type):
    if cfg.DEBUG_CLASS_WHOLE == True:
        return WHOLE_DEBUGER[class_type]
        
    if 1 <= class_type <= 7 \
            or 9 <= class_type <= 10 \
            or 12 <= class_type <= 19:
        # 16 class 1 type
        out_number = 1
    else:
        if class_type == 8 \
                or class_type == 11 \
                or class_type == 20:
            # three class 3 types
            out_number = 3
        else:
            # 7 class 2 type
            out_number = 2
    return out_number

class new_database(datasets.imdb):

    def __init__(self, db_name, stage, devkit_path=None):
        datasets.imdb.__init__(self, db_name)  # it is simply the name of db, no real data
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path

        # different databases is under the directory of
        # os.path.join(datasets.ROOT_DIR, 'data', 'clothesDatabase')
        # eg: /fast-rcnn/data/clothesDatabase/2015clothes
        self._data_path = os.path.join(self._devkit_path, db_name)
        self._stage = stage
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
                                 '其他条纹', '豹纹斑马纹', '格子',
                                 '圆点', '乱花', 'LOGO及印花图案', '其他纹'
                                 )
        self._neckband_classes = ('圆领', 'V领', '翻领',
                                  '立领', '高领', '围巾领',
                                  '一字领', '大翻领西装领',
                                  '连帽领', '其他领'
                                  )
        self._sleeve_classes = ('短袖', '中袖', '长袖')

        # for the three class detection, 1 for upper, 2 for lower, and
        # three for clothes that covers the whole body
        self._classes = ('__background__', 'Upper', 'Lower', 'Whole')
        if cfg.DEBUG_CLASS_WHOLE == True:
            self._classes = ('__background__', '8', '11', '20', 'test')
#        self._classes = ('__background__', '风衣', '毛呢大衣', '羊毛衫/羊绒衫',
#                         '棉服/羽绒服', '小西装/短外套',
#                         '西服', '夹克', '旗袍', '皮衣',
#                         '皮草', '婚纱', '衬衫', 'T恤',
#                         'Polo衫', '开衫', '马甲', '男女背心及吊带',
#                         '卫衣', '雪纺衫', '连衣裙', '半身裙',
#                         '打底裤', '休闲裤', '牛仔裤', '短裤',
#                         '卫裤/运动裤', '一致色', '横条纹',
#                         '纵条纹', '其他条纹', '豹纹斑马纹',
#                         '格子', '圆点', '乱花', 'LOGO及印花图案',
#                         '其他纹', '圆领', 'V领', '翻领', '立领',
#                         '高领', '围巾领', '一字领', '大翻领西装领',
#                         '连帽领', '其他领', '短袖', '中袖', '长袖'
#                         )

        self._ts_classes = ('__background__', '风衣', '毛呢大衣', '羊毛衫/羊绒衫',
                              '棉服/羽绒服',  '小西装/短外套',
                              '西服', '夹克', '旗袍', '皮衣', '皮草',
                              '婚纱', '衬衫', 'T恤', 'Polo衫', '开衫',
                              '马甲', '男女背心及吊带', '卫衣',
                              '雪纺衫', '连衣裙', '半身裙',
                              '打底裤', '休闲裤', '牛仔裤', '短裤',
                              '卫裤/运动裤'
                              )
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._ts_class_to_ind = dict(zip(self._ts_classes, xrange(len(self._ts_classes))))
        self._label_ext = '.clothInfo'

        # the name of the image, the type of the image, and the name of its
        # label file
        self._image_index, self._image_type, self._image_label, \
                self._image_twentysix_type = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb  # it is a function

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
        # print "the index: {}, the type: {}".format(len(self._image_index), len(self._image_twentysix_type))
        image_path = os.path.join(self._data_path, self._stage, str(self._image_twentysix_type[i]),
                                  str(self._image_index[i]))
        assert os.path.exists(image_path), \
            'image file does not exist: {}{}'.format(image_path, i)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        there are more than 26 sub files in the system
        use the type_classes to read all the image!
        """
        # Example path to image set file:
        # /home/wtw/fast-rcnn/data/clothesDatabase/train/1
        image_index = []
        image_type = []
        image_label = []
        image_twentysix_type = []

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
                        
        return image_index, image_type, image_label, image_twentysix_type

    def _get_default_path(self):
        """
        Return the default path of closes.
        """
        return os.path.join(datasets.ROOT_DIR, 'data')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_3CL=' + str(cfg.ThreeClass) + '_BLC=' + str(cfg.BALANCED) + '_COF=' + str(cfg.BALANCED_COF) + '_TT1000=' + str(cfg.TESTTYPE1000) + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(i)
                    for i in xrange(len(self.image_index))]  # we change the def
        print 'Finish loading annotations'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        # it is changed
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        print "The size of gt is {}".format(len(gt_roidb))
        print "The size of ss is {}".format(len(ss_roidb))

        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

        print "The ROIDB are all merged!"

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        box_list = []
        # read data from each sub file
        # use the image_index and image_type to clearify
        for i in xrange(len(self._image_index)):
            # getting the i th 
            filename = os.path.join(self._data_path, self._stage,
                    str(self._image_twentysix_type[i]), 'proposals', self._image_index[i][7:])
            assert os.path.exists(filename), \
                'Proposal box data not found at: {}'.format(filename)
            
            data = open(filename, "rb").read()
            number_proposals = struct.unpack("i", data[0:4])[0]
            number_edge = struct.unpack("i", data[4:8])[0]
            
            number_proposals = min(cfg.NUM_PPS, number_proposals)

            raw_data = np.asarray(struct.unpack(str(number_proposals * 4) + 'f',data[8: 8 + 16 * number_proposals])).reshape(number_proposals, 4)

            box_list.append(raw_data[:, :])
        
        print "The proposals are all loaded"

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                                  format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
            'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :] - 1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, i):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        if i % 100 == 0: 
            print "Now loading annotations of the {} th image".format(i)
        filename = os.path.join(self._data_path, self._stage, str(self._image_twentysix_type[i]),
                                'Label', self._image_label[i] + self._label_ext)
        # make sure the annotation is within the picture
        im = PIL.Image.open(self.image_path_at(i))
        widths = im.size[0]
        height = im.size[1]
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        type_objs = data.getElementsByTagName('clothClass')

        # nType always = 1 here, which is the case of our dataset
        nType = 0 

        for ix, obj in enumerate(type_objs):
            # add type class and texture class simultaneously
            location = obj.getElementsByTagName('Location')[0]
            validation = location.getAttributeNode('SourceQuality').childNodes[0].data
            nType = nType + 1

        num_objs = nType

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        # first we load the type class and the texture class
        for ix, obj in enumerate(type_objs):
            # add type class and texture class simultaneously
            dx = nType

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

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
