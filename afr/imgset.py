# Container for training set meta data. For working with the non-linear manifold assumption.


import json
import os
import re
from stat import FILE_ATTRIBUTE_HIDDEN

import numpy as np
from sklearn.cluster import KMeans
from .img import TImg, TImgEncoder, as_timg
from .consts import *
from .imio import imwrite
from .pathreset import pathreset
from .pca import pca, ptow
from .cmc import cmc
from .knn import knn


class ImgSet:
    def __init__(self, name, width, height, ipfx, isfx, nofss, timg_dir, cache_dir, class_names=None, refresh_cache=False):
        self.name = name
        self.width = width
        self.height = height
        self.ipfx = ipfx
        self.isfx = isfx
        self.timg_dir = timg_dir
        self.cache_dir = cache_dir
        if class_names:
            self.class_names = class_names
        if refresh_cache:
            self.clear_cache()
        self.nofss = nofss
    
    @property
    def nofss(self):
        return self._nofss
    
    @nofss.setter
    def nofss(self, x):
        self._nofss = x
        self.set_timgs()
    
    @pathreset
    def set_timgs(self):
        try:
            # if this throws an exception, then cwd will have changed to self.cache_dir
            self.timgs = self.read_timgs()
            # success implies all cache files exist for the current nofss value
        except:
            self.timgs = self.import_timgs()
            self.build_cache()
    
    
    # USEFUL ATTRIBUTES
    
    @property
    def cache_tag(self):
        # for use with read/write methods
        return f'{self.name}{TAG_NOFSS}{self.nofss}'
    
    @property
    def rmatrix(self):
        # in reality a 2D list of image pixel values
        return [timg.pixels for timg in self.timgs]
    
    @property
    def subsets(self):
        # a list of lists
        # inner list i contains all timgs that make up subset i
        subsets = [[] for i in range(self.nofss)]
        for timg in self.timgs:
            subsets[timg.ssindex].append(timg)
        return subsets
    
    @property
    def ss_sizes(self):
        return [len(ss) for ss in self.subsets]
    
    
    # DB CONSTRUCTION
    
    def get_cindex(self, fn):
        match = re.search(self.ipfx + '0*([0-9]+)' + self.isfx, fn)
        return int(match.group(1))
    
    def import_timgs(self):
        return [
            TImg(
                os.path.join(self.timg_dir, fn),
                self.get_cindex(fn))
            for fn in os.listdir(self.timg_dir)
            if not bool(os.stat(os.path.join(self.timg_dir, fn)).st_file_attributes & FILE_ATTRIBUTE_HIDDEN)]
    
    def kmeans(self):
        # sets the ssindex attribute of each timg
        # i.e. associates each timg with a subset
        if self.nofss == 1:
            for timg in self.timgs:
                timg.ssindex = 0
        else:
            rmatrix = self.rmatrix
            km = KMeans(
                n_clusters=self.nofss
            ).fit(rmatrix)
            for i, ssindex in enumerate(km.predict(rmatrix)):
                self.timgs[i].ssindex = int(ssindex)
    
    def build_cache(self):
        self.kmeans()
        for ss in self.subsets:
            # pca
            ssrmatrix = [timg.pixels for timg in ss]
            mean, eigvs, eigfs = pca(ssrmatrix)
            self.write_mean(mean, ss[0].ssindex)
            self.write_eigvs(eigvs, ss[0].ssindex)
            self.write_eigfs(eigfs, ss[0].ssindex)
            # eigenfaces
            for timg in ss:
                timg.weights = ptow(timg.pixels, mean, eigfs).tolist()
        self.write_timgs()
    
    @pathreset
    def clear_cache(self):
        os.chdir(self.cache_dir)
        for fn in os.listdir():
            if self.name in fn and fn.split('.')[-1] in ('json', 'npy', 'log'):
                os.remove(fn)
    
    
    # CLASSIFICATION
    
    def ss_by_dist(self, img, incl_dists=False):
        # returns a list of subset indices
        # subsets are sorted by distance of their means to img, from closest to farthest
        dists = [
            (ssindex, np.linalg.norm(self.read_mean(ssindex) - img.pixels))
            for ssindex in range(self.nofss)]
        dists.sort(key=lambda x: x[1])
        if incl_dists:
            return dists
        return [x[0] for x in dists]
    
    def itow(self, img, ssindex):
        # img to weights
        mean = self.read_mean(ssindex)
        eigfs = self.read_eigfs(ssindex)
        return ptow(img.pixels, mean, eigfs)
    
    def classify(self, img, f, *args, **kwargs):
        ssindex = self.ss_by_dist(img)[0]
        weights = self.itow(img, ssindex)
        subset = self.subsets[ssindex]
        match = f(weights, subset, *args, **kwargs)
        try:
            return match, self.class_names[match]
        except:
            return match, ''
    
    def cmc(self, img):
        return self.classify(img, cmc)
    
    def knn(self, img, k):
        return self.classify(img, knn, k)
    
    
    # CACHE READ/WRITE
    
    @pathreset
    def read_timgs(self):
        os.chdir(self.cache_dir)
        with open(f'{self.cache_tag}{TAG_TIMGS}.json') as file:
            timgs = json.load(file, object_hook=as_timg)
        return timgs
    @pathreset
    def write_timgs(self):
        os.chdir(self.cache_dir)
        with open(f'{self.cache_tag}{TAG_TIMGS}.json', 'w') as file:
            json.dump(self.timgs, file, cls=TImgEncoder, separators=(',', ':'))
    
    @pathreset
    def read_mean(self, ssindex):
        os.chdir(self.cache_dir)
        return np.load(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_MEAN}.npy')
    @pathreset
    def write_mean(self, mean, ssindex):
        os.chdir(self.cache_dir)
        np.save(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_MEAN}', mean)
    
    @pathreset
    def read_eigvs(self, ssindex):
        os.chdir(self.cache_dir)
        return np.load(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_EIGVS}.npy')
    @pathreset
    def write_eigvs(self, eigvs, ssindex):
        os.chdir(self.cache_dir)
        np.save(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_EIGVS}', eigvs)
    
    @pathreset
    def read_eigfs(self, ssindex):
        os.chdir(self.cache_dir)
        return np.load(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_EIGFS}.npy')
    @pathreset
    def write_eigfs(self, eigfs, ssindex):
        os.chdir(self.cache_dir)
        np.save(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_EIGFS}', eigfs)
    
    
    # IMAGE REMAKE
    
    # @pathreset
    # def rmk_img(self, img, ssindex, rmk_dir):
        # fweights = self.itow(img, ssindex)
        # mean = self.read_mean(ssindex)
        # eigfs = self.read_eigfs(ssindex)
        # rmk = eigfs.dot(fweights) + mean
        
        # fn = os.path.split(img.fp)[1]
        # temp = fn.split('.')
        # newfn, ext = '.'.join(temp[:-1]), temp[-1]
        
        # os.chdir(rmk_dir)
        # imwrite(f'{newfn}{TAG_RMK}.{ext}', rmk, self.width, self.height)
    
    # @pathreset
    # def rmk_mean(self, ssindex, rmk_dir):
        # mean = self.read_mean(ssindex)
        # os.chdir(rmk_dir)
        # imwrite(f'{self.cache_tag}{TAG_SS}{ssindex}{TAG_MEAN}.png', mean, self.width, self.height)
    
    
    # DEBUG/LOGGING
    
    def c_by_dist(self, img, ssindex, incl_dists=False):
        # returns a list of class indices sorted by closest to farthest from img's projection onto the subset with index ssindex
        weights = self.itow(img, ssindex)
        dists = [
            (timg.cindex, np.linalg.norm(timg.weights - weights))
            for timg in self.timgs
            if timg.ssindex == ssindex]
        dists.sort(key=lambda x: x[1])
        if incl_dists:
            return dists
        return [x[0] for x in dists]
    
    def log_img_cm(self, img):
        from .cmc import get_cmeans, dtocm
        
        # image name
        data = f'{str(img)}\n'
        
        # subsets and distances
        data += 'Subsets'.ljust(LJ) + 'Distance\n'
        ssdists = self.ss_by_dist(img, True)
        for i, d in ssdists:
            data += (' ' + str(i)).ljust(LJ) + (' ' + str(d)) + '\n'
        
        # for each subset
        for i, d in ssdists:
            # class means and distances
            data += f'\n<Subset {i}>\n'
            data += 'Class'.ljust(LJ) + 'Distance\n'
            dists = list(
                dtocm(
                    self.itow(img, i),
                    get_cmeans(self.subsets[i])
                ).items()
            )
            dists.sort(key=lambda x: x[1])
            for j, c in dists:
                data += (' ' + self.class_names[j]).ljust(LJ) + (' ' + str(c)) + '\n'
        
        return data
    
    @pathreset
    def log_cm(self, imgs):
        os.chdir(self.cache_dir)
        cm = [self.log_img_cm(img) for img in imgs]
        with open(f'{self.cache_tag}{TAG_CM}.log', 'w') as file:
            file.write((f'\n{50*"*"}\n\n').join(cm))
    
    def log_img_nn(self, img):
        # image name
        data = f'{str(img)}\n'
        
        # subsets and distances
        data += 'Subsets'.ljust(LJ) + 'Distance\n'
        ssdists = self.ss_by_dist(img, True)
        for i, d in ssdists:
            data += (' ' + str(i)).ljust(LJ) + (' ' + str(d)) + '\n'
        
        # for each subset
        for i, d in ssdists:
            # training images and distances
            data += f'\n<Subset {i}>\n'
            data += 'Class'.ljust(LJ) + 'Distance\n'
            for j, c in self.c_by_dist(img, i, True):
                data += (' ' + self.class_names[j]).ljust(LJ) + (' ' + str(c)) + '\n'
        
        return data
    
    @pathreset
    def log_nn(self, imgs):
        os.chdir(self.cache_dir)
        nn = [self.log_img_nn(img) for img in imgs]
        with open(f'{self.cache_tag}{TAG_NN}.log', 'w') as file:
            file.write((f'\n{50*"*"}\n\n').join(nn))