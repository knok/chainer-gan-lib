# -*- coding: utf-8 -*-

import os
import numpy as np
import chainer
from chainercv.utils import read_image

class CIFAR10Like(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.width = 32
        self.height = 32
        filenames = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".png") or fname.endswith('.jpg'):
                fullpath_fname = os.path.join(data_dir, fname)
                filenames.append(fullpath_fname)
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def get_example(self, i):
        if i > len(self):
            raise IndexError('index is too large')
        fname = self.filenames[i]
        img = np.asarray(read_image(fname, color=True), dtype=np.float32)
        img -= 127
        img /= 128
        return img

class NPZColorDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir = None, npz=None):
        self.data_dir = data_dir
        self.width = 32
        self.height = 32

        if npz is not None:
            self.images = np.load(npz)['arr_0']
            self.npz_fname = npz
        else:
            if data_dir is None:
                raise Error("data_dir is None")
            images = []
            for fname in os.listdir(data_dir):
                if fname.endswith(".png") or fname.endswith('.jpg'):
                    fullpath_fname = os.path.join(data_dir, fname)
                    img = np.asarray(read_image(fullpath_fname, color=True),
                                     dtype=np.float32)
                    img -= 128
                    img /= 128
                    images.append(img)
                self.images = np.asarray(images)

    def __len__(self):
        return self.images.shape[0]

    def get_example(self, i):
        if i > len(self):
            raise IndexError('index is too large')
        return self.images[i]

    def save_images(self, fname):
        np.savez(fname, self.images)                

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir')
    p.add_argument('output_fname', default='images.npz')
    args = p.parse_args()
    imgs = NPZColorDataset(args.data_dir)
    imgs.save_images(args.output_fname)

