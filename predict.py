#!/usr/bin/env python3

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
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
#parser.add_argument('--export-pngs', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=16)
opts = parser.parse_args()

# test data reader
test_imgs, test_filenames = data.img_filename_iterator(base_dir=opts.image_dir)

with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(test_imgs,
                      is_training=False,
                      use_skip_connections=not opts.no_use_skip_connections,
                      base_filter_size=opts.base_filter_size,
                      use_batch_norm=not opts.no_use_batch_norm)  
sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

if opts.output_label_db:
  db = LabelDB(label_db_file=opts.output_label_db)
  db.create_if_required()
else:
  db = None
  
# TODO: make this batched to speed it up for larger runs

for idx in itertools.count():
  try:
    fn, o = sess.run([test_filenames, model.output])

    #img = img[0]
    prediction = o[0]

    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)

    print("\t".join(map(str, [idx, fn[0], len(centroids)])))

    if False: #opts.export_pngs:
      # turn these into a bitmap
      # recall! centroids are in half res
      h, w, _ = img.shape
      centroids_bitmap = u.bitmap_from_centroids(centroids, h, w)
      # save rgb / bitmap side by side
#      debug_img = u.side_by_side(rgb=img, bitmap=prediction) #centroids_bitmap)
      debug_img = u.side_by_side(rgb=img, bitmap=centroids_bitmap)
      debug_img.save("predict_example_%03d.png" % idx)

    # set new labels
    if db:
      db.set_labels(fn[0], centroids, flip=True)
  
  except tf.errors.OutOfRangeError:
    # end of iterator
    break


