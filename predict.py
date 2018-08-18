#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
from label_db import LabelDB
import argparse
import itertools
import model
import numpy as np
import os
import random
import tensorflow as tf
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--num', type=int, default=None,
                    help='if set run prediction for this many random images. if not set run for all')
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--run', type=str, required=True, help='model, also used as subdir for export-pngs')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--export-pngs', default='',
                    help='how, if at all, to export pngs {"", "predictions", "centroids"}')
parser.add_argument('--base-filter-size', type=int, default=8)
parser.add_argument('--connected-components-threshold', type=float, default=0.05)
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
opts = parser.parse_args()

# feed data through an explicit placeholder to avoid using tf.data
imgs = tf.placeholder(dtype=tf.float32, shape=(1, opts.height, opts.width, 3), name='input_imgs')

# restore model
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

if opts.export_pngs:
  export_dir = "predict_examples/%s" % opts.run
  print("exporting prediction samples to [%s]" % export_dir)
  if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# TODO: make this batched to speed it up for larger runs

imgs = os.listdir(opts.image_dir)
if opts.num is not None:
  assert opts.num > 0
  imgs = random.sample(imgs, opts.num)

for idx, filename in enumerate(sorted(imgs)):

  # load next image (and add dummy batch dimension)
  img = np.array(Image.open(opts.image_dir+"/"+filename))  # uint8 0->255
  img = img.astype(np.float32)
  img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

  try:
    # run single image through model
    prediction = sess.run(model.output, feed_dict={model.imgs: [img]})[0]

    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction,
                                                    rescale=2.0,
                                                    threshold=opts.connected_components_threshold)

    print("\t".join(map(str, [idx, filename, len(centroids)])))

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
