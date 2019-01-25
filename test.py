#!/usr/bin/env python3

# given a directory of images and labels output overall P/R/F1 for entire set

import numpy as np
import util as u
from tensorflow.keras.models import load_model
from label_db import LabelDB
import os
from PIL import Image
from scipy.special import expit

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image-dir', type=str, required=True)
  parser.add_argument('--label-db', type=str, required=True)
  parser.add_argument('--run', type=str, required=True, help='model')
#  parser.add_argument('--batch-size', type=int, default=1)
#  parser.add_argument('--no-use-skip-connections', action='store_true')
#  parser.add_argument('--no-use-batch-norm', action='store_true')
#  parser.add_argument('--base-filter-size', type=int, default=8)
  parser.add_argument('--width', type=int, default=768, help='input image width')
  parser.add_argument('--height', type=int, default=1024, help='input image height')
  parser.add_argument('--connected-components-threshold', type=float, default=0.05)
  opts = parser.parse_args()
  print(opts)

  # load the model
  latest_model = "ckpts/%s/%s" % (opts.run, u.last_file_in_dir("ckpts/%s" % opts.run))
  print("using model [%s]" % latest_model)
  model = load_model(latest_model, custom_objects={'weighted_xent': u.weighted_xent})
  print(model.summary())

  label_db = LabelDB(label_db_file=opts.label_db)

  set_comparison = u.SetComparison()

  for idx, filename in enumerate(sorted(os.listdir(opts.image_dir))):
    print("filename", filename)

    # load next image
    # TODO: this block used in various places
    img = np.array(Image.open(opts.image_dir+"/"+filename))  # uint8 0->255  (H, W)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py
    print("img", img.shape)

    # run through model
    prediction = expit(model.predict(np.expand_dims(img, 0))[0])

    # calc [(x,y), ...] centroids
    predicted_centroids = u.centroids_of_connected_components(prediction,
                                                              rescale=2.0,
                                                              threshold=opts.connected_components_threshold)
    print("predicted_centroids", predicted_centroids)

    # compare to true labels
    true_centroids = label_db.get_labels(filename)
    true_centroids = [(y, x) for (x, y) in true_centroids]  # sigh...
    print("true_centroids     ", true_centroids)
    tp, fn, fp = set_comparison.compare_sets(true_centroids, predicted_centroids)
    print("tp, fn, fp", tp, fn, fp)

  precision, recall, f1 = set_comparison.precision_recall_f1()
  print("precision, recall, f1", precision, recall, f1)
