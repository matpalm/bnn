#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
from label_db import LabelDB
import argparse
import itertools
import numpy as np
import os
import sys
import tensorflow as tf
import time
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--graph', type=str, default='bnn_graph.predict.frozen.pb', help='graph.pb to use')
parser.add_argument('--export-pngs', default='',
                    help='how, if at all, to export pngs {"", "predictions", "centroids"}')
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

prediction_times = []
centroid_calc_times = []

sess = tf.Session()
for idx, filename in enumerate(os.listdir(opts.image_dir)):

  # load next image (and add dummy batch dimension)
  img = np.array(Image.open(opts.image_dir+"/"+filename))  # unit8 0->255
  img = img.astype(np.float32)
  img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

  try:
    # run single image through model
    s = time.time()
    prediction = sess.run(model_output, feed_dict={imgs_placeholder: [img]})[0]
    prediction_time = time.time() - s
    prediction_times.append(prediction_time)

    # calc [(x,y), ...] centroids
    s = time.time()
    centroids = u.centroids_of_connected_components(prediction, rescale=2.0)
    centroid_calc_time = time.time() - s
    centroid_calc_times.append(centroid_calc_time)

    print("\t".join(map(str, [idx, filename, len(centroids), prediction_time, centroid_calc_time])))

    # export some debug image (if requested)
    if opts.export_pngs != '':
      if opts.export_pngs == 'predictions':
        debug_img = u.side_by_side(rgb=img, bitmap=prediction)
      elif opts.export_pngs == 'centroids':
        debug_img = u.red_dots(rgb=img, centroids=centroids)
      else:
        raise Exception("unknown --export-pngs option")
      debug_img.save("predict_example_%03d.png" % idx)

    # set new labels (if requested)
    if db:
      db.set_labels(filename, centroids, flip=True)

  except tf.errors.OutOfRangeError:
    # end of iterator
    break

print("prediction times (all) mu=%f std=%f" % (np.mean(prediction_times), np.std(prediction_times)),
      file=sys.stderr)
if len(prediction_times) > 2:
  print("prediction times [2:]  mu=%f std=%f" % (np.mean(prediction_times[2:]), np.std(prediction_times[2:])),
        file=sys.stderr)
print("centroid calc times mu=%f std=%f" % (np.mean(centroid_calc_times), np.std(centroid_calc_times)),
      file=sys.stderr)
