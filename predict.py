#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
import argparse
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
parser.add_argument('--run', type=str, required=True, help='model, also used as subdir for export-pngs')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--export-pngs', default='',
                    help='how, if at all, to export pngs {"", "predictions", "centroids"}')
parser.add_argument('--base-filter-size', type=int, default=16)
parser.add_argument('--true-label-db', type=str, default=None, help='label for true values to compare to centroids')
opts = parser.parse_args()

# feed data through an explicit placeholder to avoid using tf.data
# (i _thought_ for a bit this was the cause of the linker .os problem but it's something else...)
imgs = tf.placeholder(dtype=tf.float32, shape=(1, 1024, 768, 3), name='input_imgs')

# restore model
with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(imgs,
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

if opts.true_label_db:
  true_db = LabelDB(label_db_file=opts.true_label_db)
else:
  true_db = None

if opts.export_pngs:
  export_dir = "predict_examples_%s" % opts.run
  if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# TODO: make this batched to speed it up for larger runs

for idx, filename in enumerate(sorted(os.listdir(opts.image_dir))):

  # load next image (and add dummy batch dimension)
  img = np.array(Image.open(opts.image_dir+"/"+filename))  # uint8 0->255
  img = img.astype(np.float32)
  img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

  try:
    # run single image through model
    prediction = sess.run(model.output, feed_dict={model.imgs: [img]})[0]

    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)

    pt_set_distance = 0.0
    if true_db is not None:
      true_centroids = true_db.get_labels(filename)
      print("PREDICTED", centroids)
      print("TRUE", true_centroids)
      pt_set_distance = u.compare_sets(true_pts=true_centroids, predicted_pts=centroids)

    print("\t".join(map(str, ["X", idx, filename, pt_set_distance, len(centroids)])))

    # export some debug image (if requested)
    if opts.export_pngs != '':
      if opts.export_pngs == 'predictions':
        debug_img = u.side_by_side(rgb=img, bitmap=prediction)
      elif opts.export_pngs == 'centroids':
        debug_img = u.red_dots(rgb=img, centroids=centroids)
      else:
        raise Exception("unknown --export-pngs option")
      debug_img.save("%s/%s.png" % (export_dir, filename))

    # set new labels (if requested)
    if db:
      db.set_labels(filename, centroids, flip=True)

  except tf.errors.OutOfRangeError:
    # end of iterator
    break
