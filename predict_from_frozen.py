#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
from label_db import LabelDB
import argparse
import itertools
import numpy as np
import os
import tensorflow as tf
import time
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--graph', type=str, default='bnn_graph.predict.frozen.pb', help='graph.pb to use')
parser.add_argument('--run', type=str, required=True, help='model')
parser.add_argument('--no-use-skip-connections', action='store_true')
parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--export-pngs', action='store_true')
parser.add_argument('--base-filter-size', type=int, default=16)
opts = parser.parse_args()

# restore frozen graph
graph_def = tf.GraphDef()
with open(opts.graph, "rb") as f:
  graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name=None)

# DEBUG, for dumping all op names
#for op in tf.get_default_graph().get_operations():
#  print(op.name)

# decide input and output
imgs_placeholder = tf.get_default_graph().get_tensor_by_name('import/input_imgs:0')
model_output = tf.get_default_graph().get_tensor_by_name('import/train_test_model/output:0')

if opts.output_label_db:
  db = LabelDB(label_db_file=opts.output_label_db)
  db.create_if_required()
else:
  db = None

sess = tf.Session()
for idx, filename in enumerate(os.listdir(opts.image_dir)):

  # load next image (and add dummy batch dimension)
  img = np.array(Image.open(opts.image_dir+"/"+filename))
  original_img = img.copy()
#  img = img.astype(np.float32) / 255
  imgs = np.expand_dims(img, 0)

  try:
    # run single image through model
    s = time.time()
    prediction = sess.run(model_output, feed_dict={imgs_placeholder: imgs})[0]
    print("predict time", time.time()-s)
    
    # calc [(x,y), ...] centroids
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)

    print("\t".join(map(str, [idx, filename, len(centroids)])))

    # export debug images (if requested)
    if opts.export_pngs:
      # turn these into a bitmap
      # recall! centroids are in half res
      # TODO: for centroids could draw red dots like in label_ui
      h, w, _ = original_img.shape
      centroids_bitmap = u.bitmap_from_centroids(centroids, h, w)
      # save rgb / bitmap side by side
#      debug_img = u.side_by_side(rgb=original_img, bitmap=prediction)
      debug_img = u.side_by_side(rgb=original_img, bitmap=centroids_bitmap)
      debug_img.save("predict_example_%03d.png" % idx)

      # set new labels (if requested)
    if db:
      db.set_labels(filename, centroids, flip=True)

  except tf.errors.OutOfRangeError:
    # end of iterator
    break


