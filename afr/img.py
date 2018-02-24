# Image meta data container.


import os
import json

from .imio import imread


class Img:
    def __init__(self, fp):
        self.fp = fp
        self.pixels = imread(fp)
    
    def __str__(self):
        return f'img({os.path.split(self.fp)[1]})'
    
    def __repr__(self):
        return self.__str__()


class TImg(Img):
    # training image; for use with Imgset
    def __init__(self, fp, cindex, ssindex=None, weights=None):
        super().__init__(fp)
        self.cindex = cindex
        self.ssindex = ssindex
        self.weights = weights
    
    def __str__(self):
        return f'timg({os.path.split(self.fp)[1]})'


class TImgEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TImg):
            return {
                'fp':obj.fp,
                'cindex':obj.cindex,
                'ssindex':obj.ssindex,
                'weights':obj.weights}
        return json.JSONEncoder.default(self, obj)


def as_timg(d):
    return TImg(
        d['fp'],
        d['cindex'],
        d['ssindex'],
        d['weights'])