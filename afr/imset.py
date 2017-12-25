# Image set object; container for training set metadata.


import os
import numpy as np

from .imio import imread, imwrite
from .pathreset import pathreset
from .pca import pca
from .eigf import eigf, build_fweights
from .cmc import dtocm, identify as cmcid
from .knn import dtoa, identify as knnid


SFX_MEAN = '_mean'
SFX_EIGVS = '_eigvs'
SFX_EIGFS = '_eigfs'
SFX_CWEIGHTS = '_cweights'
SFX_CMEANS = '_cmeans'


class Imset:
    def __init__(self, name, width, height, ipfx, isfx, ifirst, ifinal, dir_to_ims, dir_to_npy, classid=None):
        self.name = name
        self.width = width
        self.height = height
        self.ipfx = ipfx
        self.isfx = isfx
        self.ifirst = ifirst
        self.ifinal = ifinal
        self.dir_to_ims = dir_to_ims
        self.dir_to_npy = dir_to_npy
        if classid:
            self.classid = classid
    
    @property
    def nofc(self):
        # number of classes
        return self.ifinal - self.ifirst + 1
    
    def fnid(self, i):
        # images with this string pattern in their filename belongs to class i
        return f'{self.ipfx}0*{i}{self.isfx}'
    
    
    # main operations
    
    def buildnpy(self):
        pca(self)
        eigf(self)
    
    @pathreset
    def clearnpy(self):
        # remove all npy files associated with an imset
        os.chdir(self.dir_to_npy)
        filenames = [
            filename
            for filename in os.listdir()
            if self.name in filename and filename[-4:] == '.npy']
        for filename in filenames:
            os.remove(filename)
    
    def ftow(self, imfilepath):
        # import a face image and convert it to eigenface weights
        face = imread(imfilepath)
        mean = self.readmean()
        eigfs = self.readeigfs()
        return build_fweights(face, mean, eigfs)
    
    def cmc(self, imfilepath, dim=0):
        fweights = self.ftow(imfilepath)
        dists = dtocm(fweights, self, dim)
        return cmcid(dists)
    
    def knn(self, imfilepath, k, dim=0):
        fweights = self.ftow(imfilepath)
        tdists = dtoa(fweights, self, dim)
        return knnid(tdists, k)
    
    
    # npy reading and writing
    
    @pathreset
    def readmean(self):
        os.chdir(self.dir_to_npy)
        return np.load(f'{self.name}{SFX_MEAN}.npy')
    @pathreset
    def writemean(self, mean):
        os.chdir(self.dir_to_npy)
        np.save(f'{self.name}{SFX_MEAN}', mean)
    
    @pathreset
    def readeigvs(self):
        os.chdir(self.dir_to_npy)
        return np.load(f'{self.name}{SFX_EIGVS}.npy')
    @pathreset
    def writeeigvs(self, eigvs):
        os.chdir(self.dir_to_npy)
        np.save(f'{self.name}{SFX_EIGVS}', eigvs)
    
    @pathreset
    def readeigfs(self):
        os.chdir(self.dir_to_npy)
        return np.load(f'{self.name}{SFX_EIGFS}.npy')
    @pathreset
    def writeeigfs(self, eigfs):
        os.chdir(self.dir_to_npy)
        np.save(f'{self.name}{SFX_EIGFS}', eigfs)
    
    @pathreset
    def readcweights(self, i):
        os.chdir(self.dir_to_npy)
        return np.load(f'{self.name}{SFX_CWEIGHTS}{i}.npy')
    @pathreset
    def writecweights(self, i, cweights):
        os.chdir(self.dir_to_npy)
        np.save(f'{self.name}{SFX_CWEIGHTS}{i}', cweights)
    
    @pathreset
    def readcmeans(self):
        os.chdir(self.dir_to_npy)
        return np.load(f'{self.name}{SFX_CMEANS}.npy')
    @pathreset
    def writecmeans(self, cmeans):
        os.chdir(self.dir_to_npy)
        np.save(f'{self.name}{SFX_CMEANS}', cmeans)
    
    
    # image remake operations
    
    @pathreset
    def rmkim(self, imfilepath, dir_to_rmk):
        # remake an arbitrary image
        filename = os.path.split(imfilepath)[1]
        fweights = self.ftow(imfilepath)
        mean = self.readmean()
        eigfs = self.readeigfs()
        face = eigfs.dot(fweights) + mean
        os.chdir(dir_to_rmk)
        imwrite(f'rmk_{filename}', face, self.width, self.height)
    
    @pathreset
    def rmkmean(self, dir_to_rmk):
        # export the mean as an image
        mean = self.readmean()
        os.chdir(dir_to_rmk)
        imwrite(f'{self.name}_mean.png', mean, self.width, self.height)
    
    @pathreset
    def rmkeigfs(self, dir_to_rmk):
        # export all eigenfaces for human-viewing
        mean = self.readmean()
        eigfs = self.readeigfs()
        os.chdir(dir_to_rmk)
        for j in range(eigfs.shape[1]):
            # 100 was arbitrarily chosen such that it gave a nice-looking result
            c = 100 / max(eigfs[:,j], key=lambda x:abs(x))
            imwrite(f'{self.name}_eigf{j}.png', c*eigfs[:,j] + mean, self.width, self.height)
    
    @pathreset
    def rmkcmeans(self, dir_to_rmk):
        # exports the average of each class
        mean = self.readmean()
        eigfs = self.readeigfs()
        cmeans = self.readcmeans()
        os.chdir(dir_to_rmk)
        for j in range(cmeans.shape[1]):
            face = eigfs.dot(cmeans[:,j]) + mean
            imwrite(f'{self.name}_cmean{j}.png', face, self.width, self.height)