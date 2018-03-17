# Container for training set meta data. For working with the non-linear manifold assumption.


import json
import os
import re
from stat import FILE_ATTRIBUTE_HIDDEN

import numpy as np
from sklearn.cluster import KMeans
from .img import TImg, TImgEncoder, as_timg
from .consts import *
from .imio import imread, imwrite
from .pathreset import pathreset
from .pca import pca, ptow
from .cmc import cmc
from .knn import knn
from .timer import timer

# for logging methods
from .cmc import get_cmeans, dtocm
from .tabulate import tabulate



class ImgSet:
    def __init__(self, name, width, height, ipfx, isfx, nofss, timg_dir, db_dir, class_names=None, refresh_db=False):
        self.name = name
        self.width = width
        self.height = height
        self.ipfx = ipfx
        self.isfx = isfx
        self.nofss = nofss
        self.timg_dir = timg_dir
        self.db_dir = db_dir
        self.class_names = class_names
        if refresh_db:
            self.clear()
    
    
    # USEFUL ATTRIBUTES
    
    @property
    def db_tag(self):
        # for use with read/write methods
        return f'{self.name}{TAG_NOFSS}{self.nofss}'
    
    @property
    def dim(self):
        return self.width*self.height
    
    @property
    def rmatrix(self):
        # actually a list of float64 arrays
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
        timgs = []
        for fn in os.listdir(self.timg_dir):
            if not bool(os.stat(os.path.join(self.timg_dir, fn)).st_file_attributes & FILE_ATTRIBUTE_HIDDEN):
                timg = TImg(
                    os.path.join(self.timg_dir, fn),
                    self.get_cindex(fn)
                )
            if len(timg.pixels) != self.dim:
                raise IndexError(f'dimensions for {timg} and image set do not match')
            timgs.append(timg)
        return timgs
    
    @timer(TIME_KMEANS)
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
    
    @timer(TIME_BUILD)
    @pathreset
    def build(self):
        try:
            # if this throws an exception, then cwd will have changed to self.db_dir
            self.timgs = self.read_timgs()
            # success implies all db files exist for the current nofss value
        except:
            self.timgs = self.import_timgs()
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
                    if SHOW_WEIGHTS_PROGRESS:
                        print(timg.fn)
                    timg.weights = ptow(timg.pixels, mean, eigfs)
            self.write_timgs()
    
    @pathreset
    def clear(self):
        os.chdir(self.db_dir)
        for fn in os.listdir():
            if re.search(f'{self.name}_.*[.](json|npy|log)', fn, re.I):
                os.remove(fn)
    
    
    # CLASSIFICATION
    
    def ss_by_dist(self, img, incl_dists=False):
        # returns a list of subset indices
        # subsets are sorted by distance of their means to img, from closest to farthest
        dists = [
            [ssindex, np.linalg.norm(self.read_mean(ssindex) - img.pixels)]
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
    
    def classify(self, img, func, j, *args, **kwargs):
        # project image onto the j-nearest subsets and do CMC or KNN on each
        # each of the j subsets will produce a candidate
        # if using CMC or 1NN, return the candidate with the smallest distance to img
        # if using KNN with k>1, return the candidate with the smallest sum(distances of candidate's training images to img)
        if len(img.pixels) != self.dim:
            raise IndexError('dimensions for image and image set do not match')
        ssindices = self.ss_by_dist(img)[:j]
        subsets = self.subsets
        # list of (class index, distance) tuples
        candidates = []
        for ssindex in ssindices:
            weights = self.itow(img, ssindex)
            candidates.append(func(weights, subsets[ssindex], *args, **kwargs))
        match = min(candidates, key=lambda x: x[1])[0]
        try:
            return match, self.class_names[match]
        except:
            return match, ''
    
    def cmc(self, img, j=1):
        return self.classify(img, cmc, j)
    
    def knn(self, img, j=1, k=1):
        return self.classify(img, knn, j, k)
    
    # CACHE READ/WRITE
    
    @pathreset
    def read_timgs(self):
        os.chdir(self.db_dir)
        with open(f'{self.db_tag}{TAG_TIMGS}.json') as file:
            timgs = json.load(file, object_hook=as_timg)
        return timgs
    @pathreset
    def write_timgs(self):
        os.chdir(self.db_dir)
        with open(f'{self.db_tag}{TAG_TIMGS}.json', 'w') as file:
            json.dump(self.timgs, file, cls=TImgEncoder, separators=(',', ':'))
    
    @pathreset
    def read_mean(self, ssindex):
        os.chdir(self.db_dir)
        return np.load(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_MEAN}.npy')
    @pathreset
    def write_mean(self, mean, ssindex):
        os.chdir(self.db_dir)
        np.save(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_MEAN}', mean)
    
    @pathreset
    def read_eigvs(self, ssindex):
        os.chdir(self.db_dir)
        return np.load(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_EIGVS}.npy')
    @pathreset
    def write_eigvs(self, eigvs, ssindex):
        os.chdir(self.db_dir)
        np.save(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_EIGVS}', eigvs)
    
    @pathreset
    def read_eigfs(self, ssindex):
        os.chdir(self.db_dir)
        return np.load(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_EIGFS}.npy')
    @pathreset
    def write_eigfs(self, eigfs, ssindex):
        os.chdir(self.db_dir)
        np.save(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_EIGFS}', eigfs)
    
    
    # IMAGE REMAKE
    
    @pathreset
    def rmk_img(self, img, rmk_dir, ssindex=0):
        fweights = self.itow(img, ssindex)
        mean = self.read_mean(ssindex)
        eigfs = self.read_eigfs(ssindex)
        rmk = eigfs @ fweights + mean
        
        fn = os.path.split(img.fp)[1]
        temp = fn.split('.')
        newfn, ext = '.'.join(temp[:-1]), temp[-1]
        
        os.chdir(rmk_dir)
        imwrite(f'{newfn}{TAG_RMK}.{ext}', rmk, self.width, self.height)
    
    @pathreset
    def rmk_mean(self, rmk_dir, ssindex=0):
        mean = self.read_mean(ssindex)
        os.chdir(rmk_dir)
        imwrite(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_MEAN}.png', mean, self.width, self.height)
    
    @pathreset
    def rmk_eigfs(self, rmk_dir, ssindex=0):
        eigfs = self.read_eigfs(ssindex)
        os.chdir(rmk_dir)
        for j in range(eigfs.shape[1]):
            imwrite(f'{self.db_tag}{TAG_SS}{ssindex}{TAG_EIGF}{j}.png', eigfs[:,j], self.width, self.height)
    
    
    # DEBUG/LOGGING
    
    def ti_by_dist(self, img, ssindex):
        # returns a list of (timg.cindex, distance between img and timg, timg.fn) tuples for when img is projected onto a given subset
        weights = self.itow(img, ssindex)
        dists = [
            [
                timg.cindex,
                np.linalg.norm(timg.weights - weights),
                timg.fn
            ]
            for timg in self.subsets[ssindex]
        ]
        dists.sort(key=lambda x: x[1])
        return dists
    
    def format_img(self, img, mode, margin=2):
        ssdists = self.ss_by_dist(img, True)
        
        # tabulate distance to subset means
        tables = tabulate(
            [['SUBSET', 'DISTANCE']] + ssdists,
            str(img),
            margin
        )
        
        for i, d in ssdists:
            # for each subset tabulate either distance to class means or distance to training images
            if mode == 'cm':
                headings = ['CLASS', 'DISTANCE']
                dists = list(
                    dtocm(
                        self.itow(img, i),
                        get_cmeans(self.subsets[i])
                    ).items()
                )
                dists.sort(key=lambda x: x[1])
                for j in range(len(dists)):
                    dists[j] = list(dists[j]) # convert tuple to list
            elif mode == 'ti':
                headings = ['CLASS', 'DISTANCE', 'FILENAME']
                dists = self.ti_by_dist(img, i)
            
            if self.class_names:
                headings.insert(1, 'NAME')
                for item in dists:
                    item.insert(1, self.class_names[item[0]])
            
            tables += tabulate(
                [headings] + dists,
                f'<Subset {i}>',
                margin
            )
        
        return tables
    
    @pathreset
    def log_imgs(self, imgs, mode):
        all_tables = [self.format_img(img, mode) for img in imgs]
        if mode == 'cm':
            tag = TAG_CM
        elif mode == 'ti':
            tag = TAG_TI
        os.chdir(self.db_dir)
        with open(f'{self.db_tag}{tag}.log', 'w') as file:
            file.write((f'\n{AWIDTH*"*"}\n\n').join(all_tables))