import sys
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import threading

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def crop2(x, y, wrg=96, hrg=96, is_random=True, row_index=0, col_index=1):
    h, w = y.shape[row_index], y.shape[col_index]

    if (h <= hrg) or (w <= wrg):
        raise AssertionError("The size of cropping should smaller than the original image")
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
    else:  # central crop
        h_offset = int(np.floor((h - hrg) / 2.))
        w_offset = int(np.floor((w - wrg) / 2.))
    newx,newy=(x[h_offset*4:hrg*4 + h_offset*4, w_offset*4:wrg*4 + w_offset*4],y[h_offset:hrg + h_offset, w_offset:wrg + w_offset])
    newx = newx / (255. / 2.) - 1.
    newy = newy / (255. / 2.) - 1.
    return (newx,newy)

def threading_data_2(data=None, fn=None, thread_count=None, **kwargs):
    hr,lr = data
    def apply_fn(results, i, data_hr, data_lr, kwargs):
        results[i]=fn(data_hr, data_lr, **kwargs)

    results = [None] * len(data[0])
    threads = []
    for i in range(len(hr)):
        t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, hr[i],lr[i], kwargs))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if thread_count is None:
        try:
            # results=[i[0] for i in results]
            return np.asarray([i[0] for i in results]),np.asarray([i[1] for i in results])
        except Exception:
            print("error")
            return np.concatenate(results[0]),np.concatenate(results[1])


if __name__ == '__main__':
    pass
