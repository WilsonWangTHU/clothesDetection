# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps, bbox_coverage
import numpy as np
import scipy.sparse
import datasets
from fast_rcnn.config import cfg

def twentysix2three(class_type):
    if cfg.DEBUG_CLASS_WHOLE == True:
        return WHOLE_DEBUGER[class_type]
        
    if 1 <= class_type <= 7 \
            or 9 <= class_type <= 10 \
            or 12 <= class_type <= 19:
        # 17 class 1 type
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


class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._len_label = 0
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def len_label(self):
        return self._len_label

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def ts_classes(self):
        return self._ts_classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    @property
    def image_twentysix_type(self):
        return self._image_twentysix_type
        
    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        # I find something big! the PIL open has mistake!,making 
		# width = 349
        # widths = [PIL.Image.open(self.image_path_at(i)).\
        # size[0] for i in xrange(num_images)]
        widths = [350 for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all(), \
                    "The box size and width is not matched"
            
            if cfg.MULTI_LABEL == True:
                entry = {'boxes' : boxes,
                        'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                        'gt_classes' : self.roidb[i]['gt_classes'],
                        'flipped' : True,
                        'multi_label' : self.roidb[i]['multi_label']}
            else:
                entry = {'boxes' : boxes,
                        'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                        'gt_classes' : self.roidb[i]['gt_classes'],
                        'flipped' : True}
            self.roidb.append(entry)

        # edit the needed index and type variables
        self._image_index = self._image_index * 2
        self._image_twentysix_type = self._image_twentysix_type * 2
        self._image_type = self._image_type * 2

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []

        # in each images, there is a box list, i.e. the box_list[i]
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]  # this is important
            overlaps = np.zeros((num_boxes, self.num_classes),
                    dtype=np.float32)
            if cfg.MULTI_LABEL == True:
                if cfg.MULTI_LABEL_SOFTMAX == True:
                    multi_label = np.zeros((num_boxes,
                        self.len_label), dtype=np.int32)
                else:
                    multi_label = -1 * np.ones((num_boxes,
                        self.len_label), dtype=np.int32)

            if gt_roidb is not None:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                # debug                
                #gt_boxes[0,:] = [100, 100, 200, 200]
                #boxes[0, :] = [150, 200, 170, 220]
                if not cfg.BG_CHOICE:
                    gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                            gt_boxes.astype(np.float), -1,
                            cfg.BG_VALID_THRESH)
                else:
                    gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                            gt_boxes.astype(np.float), 
                            int(twentysix2three(self.image_twentysix_type[i])),
                            cfg.BG_VALID_THRESH) # twentysix2three(self.ts_classes[i])

                argmaxes = gt_overlaps.argmax(axis=1)  # the index for the max gt
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]

                # set the max overlaped class, take out the sub max
                # the gt_overlaps is bbox * gt, while the overlaps is bbox * gt_class
                # the dim is different
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

                # now for the multi_label (attrbutive part), the idea is similar,
                # we compute the overlaps (not IOU, actually)

                sleeve_coordinate1 = gt_roidb[i]['sleeve_coordinate1']
                sleeve_coordinate2 = gt_roidb[i]['sleeve_coordinate2']
                neckband_coordinate = gt_roidb[i]['neckband_coordinate']
                
                active_sleeve1 = bbox_coverage(boxes.astype(np.float),                         
                        sleeve_coordinate1.astype(np.float))
                active_sleeve2 = bbox_coverage(boxes.astype(np.float), 
                        sleeve_coordinate2.astype(np.float))
                active_neckband = bbox_coverage(boxes.astype(np.float), 
                        neckband_coordinate.astype(np.float))
                # load the texture infomation
                if cfg.MULTI_LABEL:                    
                    multi_label[I, 0:cfg.NUM_MULTI_LABEL_TEXTURE] = \
                        gt_roidb[i]['multi_label'][argmaxes[I], \
                        0:cfg.NUM_MULTI_LABEL_TEXTURE]

                # the multi_label seems to have an error, the [i][multi_label]
                # could have multiple items!
                if cfg.MULTI_LABEL == True:
                    if not cfg.ATTR_CHOICE:
                        multi_label[I, :] = gt_roidb[i]['multi_label'][argmaxes[I]]
                    else:
                        # we must check whether the sleeve or the neckband
                        # is in the bounding box, IOU > 0.5?
                        argmaxes = active_sleeve1.argmax(axis=1)
                        maxes = active_sleeve1.max(axis=1)
                        I = np.where(maxes > cfg.ATTR_THRESH)[0]
                        multi_label[I, cfg.NUM_MULTI_LABEL_NECKBAND + \
                                cfg.NUM_MULTI_LABEL_TEXTURE:] = \
                                gt_roidb[i]['multi_label'][argmaxes[I], \
                                cfg.NUM_MULTI_LABEL_NECKBAND + \
                                cfg.NUM_MULTI_LABEL_TEXTURE:]

                        argmaxes = active_sleeve2.argmax(axis=1)
                        maxes = active_sleeve2.max(axis=1)
                        I = np.where(maxes > cfg.ATTR_THRESH)[0]
                        multi_label[I, cfg.NUM_MULTI_LABEL_NECKBAND + \
                                cfg.NUM_MULTI_LABEL_TEXTURE:] = \
                                gt_roidb[i]['multi_label'][argmaxes[I], \
                                cfg.NUM_MULTI_LABEL_NECKBAND + \
                                cfg.NUM_MULTI_LABEL_TEXTURE:]

                        argmaxes = active_neckband.argmax(axis=1)
                        maxes = active_neckband.max(axis=1)
                        I = np.where(maxes > cfg.ATTR_THRESH)[0]
                        multi_label[I, cfg.NUM_MULTI_LABEL_TEXTURE : \
                                cfg.NUM_MULTI_LABEL_TEXTURE + \
                                cfg.NUM_MULTI_LABEL_NECKBAND] = \
                                gt_roidb[i]['multi_label'][argmaxes[I], \
                                cfg.NUM_MULTI_LABEL_TEXTURE: \
                                cfg.NUM_MULTI_LABEL_NECKBAND + \
                                cfg.NUM_MULTI_LABEL_TEXTURE]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            if cfg.MULTI_LABEL == False:
                roidb.append({'boxes' : boxes,
                              'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_overlaps' : overlaps,
                              'flipped' : False})
            else:
                roidb.append({'boxes' : boxes,
                              'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_overlaps' : overlaps,
                              'flipped' : False,
                              'multi_label' : multi_label})
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b), \
                "The size of the gt and the fg/bg boxes are not matched?"
        for i in xrange(len(a)):
            if i % 1000 == 0:
                print('Merging the {} th image gt and boxes')

            assert a[i]['boxes'].shape[1] == b[i]['boxes'].shape[1], \
                    "Boxes size not matched! error! at {}".format(i)
            assert a[i]['gt_overlaps'].shape[1] == \
                    b[i]['gt_overlaps'].shape[1], \
                    "gt_overlaps sizes are not matched at {}".format(i)

            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            if cfg.MULTI_LABEL == True:
                a[i]['multi_label'] = \
                        np.vstack((a[i]['multi_label'], b[i]['multi_label']))
        print("We finished the merging task of gt and boxes!")
        return a
        
    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
