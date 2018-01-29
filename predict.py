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
parser.add_argument('--label-db', type=str, required=True)
parser.add_argument('--run', type=str, required=True, help='model')
opts = parser.parse_args()

# test data reader
test_imgs, test_filenames = data.img_filename_iterator(base_dir=opts.image_dir)

with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(test_imgs)
  
sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

db = LabelDB(label_db_file=opts.label_db)
db.create_if_required()

for idx in itertools.count():
  try:
    fn, logits, o = sess.run([test_filenames,
                              model.logits, model.output],
                             feed_dict={model.is_training: False})
    print("idx", idx, fn[0])
    print("logits", logits.shape, np.min(logits), np.max(logits))

    prediction = o[0]

    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)

    # turn these into a bitmap
    # recall! centroids are in half res
#    h, w, _ = img.shape
#    centroids_bitmap = u.bitmap_from_centroids(centroids, h, w)
    # save rgb / bitmap side by side
#    debug_img = u.side_by_side(rgb=img, bitmap=centroids_bitmap)
#    debug_img.save("example_%03d.png" % idx)

    # set new labels
    db.set_labels(fn[0], centroids, flip=True)
  
  except tf.errors.OutOfRangeError:
    # end of iterator
    break


