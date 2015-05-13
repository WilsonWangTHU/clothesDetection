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

''' in this file, we try to enable new database to be used'''


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
        self._classes = ('__background__', '风衣', '羊毛衫/羊绒衫',
                         '毛呢大衣', '棉服/羽绒服', '小西装/短外套',
                         '西服', '夹克', '旗袍', '皮衣',
                         '皮草', '婚纱', '衬衫', 'T恤',
                         'Polo衫', '开衫', '马甲', '男女背心及吊带',
                         '卫衣', '雪纺衫', '连衣裙', '半身裙',
                         '打底裤', '休闲裤', '牛仔裤', '短裤',
                         '卫裤/运动裤', '一致色', '横条纹',
                         '纵条纹', '其他条纹', '豹纹斑马纹',
                         '格子', '圆点', '乱花', 'LOGO及印花图案',
                         '其他纹', '圆领', 'V领', '翻领', '立领',
                         '高领', '围巾领', '一字领', '大翻领西装领',
                         '连帽领', '其他领', '短袖', '中袖', '长袖'
                         )

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._label_ext = '.clothInfo'

        # the name of the image, the type of the image, and the name of its
        # label file
        self._image_index, self._image_type, self._image_label = \
            self._load_image_set_index()
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
        image_path = os.path.join(self._data_path, self._stage, str(self._image_type[i]),
                                  'image_50',
                                  str(self._image_index[i]))
        assert os.path.exists(image_path), \
            'image file does not exist: {}'.format(image_path)
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
        for class_type in xrange(1, len(self._type_classes) + 1):
            image_set_file = os.path.join(self._data_path, self._stage, str(class_type),
                                          'GUIDMapping.txt')
            assert os.path.exists(image_set_file), \
                'index txt does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                for x in f.readlines():
                    y = x.strip()
                    if y.find('.jpg') == -1:  # it is not a jpg file
                        image_label.append(y[y.find('.png') + 1 + 4:])
                        image_index.append(y[y.find('\\') + 1: y.find('.png')] + '.png')
                    else:
                        image_label.append(y[y.find('.jpg') + 1 + 4:])
                        image_index.append(y[y.find('\\') + 1: y.find('.jpg')] + '.jpg')
                    image_type.append(class_type)

        return image_index, image_type, image_label

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
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(i)
                    for i in xrange(len(self.image_index))]  # we change the def
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
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        box_list = []
        # read data from each sub file
        for i in xrange(len(self._type_classes)):
            filename = os.path.abspath(os.path.join(datasets.ROOT_DIR, 'data',
                                       self.name, self._stage, str(i + 1), 'boxes.mat'))
            assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
            raw_data = sio.loadmat(filename)['boxes'].ravel()
            for x in xrange(raw_data.shape[0]):
                box_list.append(raw_data[x][:, (1, 0, 3, 2)] - 1)

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
        filename = os.path.join(self._data_path, self._stage, str(self._image_type[i]),
                                'Label', self._image_label[i] + self._label_ext)
        # make sure the annotation is within the picture
        im = PIL.Image.open(self.image_path_at(i))
        widths = im.size[0]
        height = im.size[1]
        if i == 167:
            print widths
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        type_objs = data.getElementsByTagName('clothClass')
        texture_objs = data.getElementsByTagName('clothClass')

        neckband_objs = data.getElementsByTagName('clothNeckband')
        sleeve_objs = data.getElementsByTagName('clothSleeve')
        num_objs = len(neckband_objs) + len(sleeve_objs) + len(texture_objs) + \
            len(type_objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        # first we load the type class and the texture class
        for ix, obj in enumerate(type_objs):
            # add type class and texture class simultaneously
            dx = len(type_objs)

            # Make pixel indexes 0-based
            location = obj.getElementsByTagName('Location')[0]
            
            validation = location.getAttributeNode('SourceQuality').childNodes[0].data
            if validation != u'Valid':
                # continue  # this box is useless, continue
                x1 = widths / 3.0
                x2 = widths / 3.0 * 2.0
                y1 = height / 3.0
                y2 = height / 3.0 * 2.0
    
                cls = self._image_type[i]  # change the unicode into str
                print cls
                print type(cls)
    
            else:
                x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
                y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
                x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
                y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))
    
                x1 = min(x1, widths - 1)
                x2 = min(x2, widths - 1)
                y1 = min(y1, height - 1)
                y2 = min(y2, height - 1)
    
                cls_str = obj.getAttributeNode('type').childNodes[0].data
                cls_str = cls_str.encode('utf-8')  # change the unicode into str
    
                cls = self._class_to_ind[cls_str]

            # add type class
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            # for the texture
            if validation != u'Valid':
                continue
            tex_cls_str = texture_objs[ix].getAttributeNode('type').childNodes[0].data
            tex_cls_str = tex_cls_str.encode('utf-8')  # change the unicode into str
            if tex_cls_str == '其他':
                tex_cls_str = '其他纹'
            tex_cls = self._class_to_ind[tex_cls_str]
            # add texture class
            boxes[ix + dx, :] = [x1, y1, x2, y2]
            gt_classes[ix + dx] = cls
            overlaps[ix + dx, tex_cls] = 1.0

        # loading the object bounding box of the sleeve type and the neckband type
        dx = len(type_objs) + len(texture_objs)
        for ix, obj in enumerate(neckband_objs):
            # Make pixel indexes 0-based
            location = obj.getElementsByTagName('Location')[0]

            x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
            y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
            x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
            y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))

            x1 = min(x1, widths - 1)
            x2 = min(x2, widths - 1)
            y1 = min(y1, height - 1)
            y2 = min(y2, height - 1)
            # check for validation
            validation = location.getAttributeNode('SourceQuality').childNodes[0].data
            if validation != u'Valid':
                continue  # this box is useless, continue

            cls_str = obj.getAttributeNode('type').childNodes[0].data
            cls_str = cls_str.encode('utf-8')  # change the unicode into str
            print cls_str
            print type(cls_str)
			
            if cls_str == '其他':  # we have to devide them!
                cls_str = '其他领'
            cls = self._class_to_ind[cls_str]

            # add type class
            boxes[ix + dx, :] = [x1, y1, x2, y2]
            gt_classes[ix + dx] = cls
            overlaps[ix + dx, cls] = 1.0

        dx = len(type_objs) + len(texture_objs) + len(neckband_objs)
        for ix, obj in enumerate(sleeve_objs):
            # check for validation

            # Make pixel indexes 0-based
            location = obj.getElementsByTagName('Location')[0]
            validation = location.getAttributeNode('SourceQuality').childNodes[0].data
            if validation != u'Valid':
                continue  # this box is useless, continue

            x1 = float(floor(float(location.getAttributeNode('left').childNodes[0].data)))
            y1 = float(floor(float(location.getAttributeNode('top').childNodes[0].data)))
            x2 = float(floor(float(location.getAttributeNode('right').childNodes[0].data)))
            y2 = float(floor(float(location.getAttributeNode('bottom').childNodes[0].data)))

            x1 = min(x1, widths - 1)
            x2 = min(x2, widths - 1)
            y1 = min(y1, height - 1)
            y2 = min(y2, height - 1)
            cls_str = obj.getAttributeNode('type').childNodes[0].data
            cls_str = cls_str.encode('utf-8')  # change the unicode into str
            cls = self._class_to_ind[cls_str]

            # add type class
            boxes[ix + dx, :] = [x1, y1, x2, y2]
            gt_classes[ix + dx] = cls
            overlaps[ix + dx, cls] = 1.0

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
