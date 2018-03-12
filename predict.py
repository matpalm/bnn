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
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--export-pngs', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=16)
opts = parser.parse_args()

# feed data through an explicit placeholder to avoid using tf.data
# (i _thought_ for a bit this was the cause of the linker .os problem but it's something else...)
imgs = tf.placeholder(dtype=tf.uint8, shape=(1, 1024, 768, 3))

# test data reader
with tf.variable_scope("train_test_model") as scope:  # clumsy :/
  model = model.Model(imgs,
                      is_training=False,
                      use_skip_connections=not opts.no_use_skip_connections,
                      base_filter_size=opts.base_filter_size,
                      use_batch_norm=not opts.no_use_batch_norm)  
sess = tf.Session()
model.restore(sess, "ckpts/%s" % opts.run)

tf.train.write_graph(sess.graph_def, ".", "bnn_graph.predict.pb")

if opts.output_label_db:
  db = LabelDB(label_db_file=opts.output_label_db)
  db.create_if_required()
else:
  db = None
  
# TODO: make this batched to speed it up for larger runs

for idx, filename in enumerate(os.listdir(opts.image_dir)):

  # load next image (and add dummy batch dimension)
  img = np.array(Image.open(opts.image_dir+"/"+filename))
  imgs = np.expand_dims(img, 0)

  try:
    # run single image through model
    prediction = sess.run(model.output, feed_dict={model.imgs: imgs})[0]

    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)

    print("\t".join(map(str, [idx, filename, len(centroids)])))

    # export debug images (if requested)
    if opts.export_pngs:
      # turn these into a bitmap
      # recall! centroids are in half res
      # TODO: for centroids could draw red dots like in label_ui
      h, w, _ = img.shape
      centroids_bitmap = u.bitmap_from_centroids(centroids, h, w)
      # save rgb / bitmap side by side
#      debug_img = u.side_by_side(rgb=img, bitmap=prediction)
      debug_img = u.side_by_side(rgb=img, bitmap=centroids_bitmap)
      debug_img.save("predict_example_%03d.png" % idx)

      # set new labels (if requested)
    if db:
      db.set_labels(filename, centroids, flip=True)
  
  except tf.errors.OutOfRangeError:
    # end of iterator
    break


