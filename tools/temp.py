# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:59:35 2015

@author: twwang
"""
import xml.dom.minidom as minidom

filename = '/media/Elements/twwang/fast-rcnn/data/clothesDataset/test/1/Label'
filename = filename + '/0a47607ad5c740d282581dafd0c07064.clothInfo'
with open(filename) as f:
    data = minidom.parseString(f.read())


_texture_classes = ('一致色', '横条纹', '纵条纹',
                                 '其他条纹', '豹纹斑马纹', '格子',
                                 '圆点', '乱花', 'LOGO及印花图案', '其他'
                                 )
textcls_ind = dict(zip(_texture_classes, xrange(5, 5 + len(_texture_classes))));
type_objs = data.getElementsByTagName('clothNeckband')
text_type = type_objs[0]
texture_cls = text_type.getAttributeNode('type').childNodes[0].data
texture_cls = texture_cls.encode('utf-8')