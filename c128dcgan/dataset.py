# -*- coding: utf-8 -*-

import os
import numpy as np
import chainer
from chainercv.utils import read_image

class Color128x128Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.width = 128
        self.height = 128

        filenames = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".png") or fname.endswith('.jpg'):
                fullpath_fname = os.path.join(data_dir, fname)
                filenames.append(fullpath_fname)
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def get_exmaple(self, i):
        if i > len(self):
            raise IndexError('index is too large')
        fname = self.filenames[i]
        img = read_image(fname, color=True)
        return img
