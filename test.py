#!/usr/bin/env python
from __future__ import print_function

# given a directory of images output a list of image -> predictions

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
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--batch-size', type=int, default=10)
opts = parser.parse_args()

# test data reader
test_imgs, test_xys_bitmaps = data.img_xys_iterator(base_dir=opts.image_dir,
                                                    batch_size=opts.batch_size,
                                                    patch_fraction=1,
                                                    distort=False,
                                                    repeat=False)

with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(test_imgs)
  
sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

# TODO: refactor out... clusmy / common code with train.py
dice_loss = u.dice_loss(tf.to_float(test_xys_bitmaps),
                        tf.sigmoid(model.logits),
                        batch_size=opts.batch_size, 
                        smoothing=1e-2)
    
for idx in itertools.count():
  try:
    dice_l = sess.run(dice_loss,
                      feed_dict={model.is_training: False})
    
    print("idx", idx)
    print("dice_loss", np.mean(dice_l), dice_l)

  except tf.errors.OutOfRangeError:
    # end of iterator
    break


