#!/usr/bin/env python3

# given a directory of images and labels output some stats

from PIL import Image, ImageDraw
import argparse
import data
import itertools
from label_db import LabelDB
import model
import numpy as np
import os
import tensorflow as tf
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--label-dir', type=str, required=True)
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=8)
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
opts = parser.parse_args()
print(opts)

# test data reader
test_imgs, test_xys_bitmaps = data.img_xys_iterator(image_dir=opts.image_dir,
                                                    label_dir=opts.label_dir,
                                                    batch_size=opts.batch_size,
                                                    patch_width_height=0,  # i.e. no patchs
                                                    distort_rgb=False,
                                                    flip_left_right=False,
                                                    random_rotation=False,
                                                    repeat=False,
                                                    width=opts.width,
                                                    height=opts.height)

model = model.Model(test_imgs,
                    is_training=False,
                    use_skip_connections=not opts.no_use_skip_connections,
                    base_filter_size=opts.base_filter_size,
                    use_batch_norm=not opts.no_use_batch_norm)
model.calculate_losses_wrt(labels=test_xys_bitmaps)

sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

xent = []
for idx in itertools.count():
  try:
    xent.append(sess.run(model.xent_loss))
  except tf.errors.OutOfRangeError:
    # end of iterator
    break

# TODO: there must be a pandas version of this...
print("      %6s %6s %6s %6s" % ("min", "max", "mean", "std"))
print("xent  %0.4f %0.4f %0.4f %0.4f" % (np.min(xent), np.max(xent), np.mean(xent), np.std(xent)))
