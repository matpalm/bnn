#!/usr/bin/env python3

# given a directory of images and labels output overall P/R/F1 for entire set

from PIL import Image
from label_db import LabelDB
from scipy.special import expit
import model as m
import numpy as np
import os
import util as u

def pr_stats(run, image_dir, label_db, connected_components_threshold):

  # TODO: a bunch of this can go back into one off init in a class

  _train_opts, model = m.restore_model(run)

  label_db = LabelDB(label_db_file=label_db)

  set_comparison = u.SetComparison()

  # use 4 images for debug
  debug_imgs = []

  for idx, filename in enumerate(sorted(os.listdir(image_dir))):
    # load next image
    # TODO: this block used in various places, refactor
    img = np.array(Image.open(image_dir+"/"+filename))  # uint8 0->255  (H, W)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

    # run through model
    prediction = expit(model.predict(np.expand_dims(img, 0))[0])

    if len(debug_imgs) < 4:
      debug_imgs.append(u.side_by_side(rgb=img, bitmap=prediction))

    # calc [(x,y), ...] centroids
    predicted_centroids = u.centroids_of_connected_components(prediction,
                                                              rescale=2.0,
                                                              threshold=connected_components_threshold)

    # compare to true labels
    true_centroids = label_db.get_labels(filename)
    true_centroids = [(y, x) for (x, y) in true_centroids]  # sigh...
    tp, fn, fp = set_comparison.compare_sets(true_centroids, predicted_centroids)

  precision, recall, f1 = set_comparison.precision_recall_f1()

  return {"debug_imgs": debug_imgs,
          "precision": precision,
          "recall": recall,
          "f1": f1}

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--run', type=str, required=True, help='model')
  parser.add_argument('--image-dir', type=str, required=True)
  parser.add_argument('--label-db', type=str, required=True)
  parser.add_argument('--connected-components-threshold', type=float, default=0.05)
  opts = parser.parse_args()
  print(opts)

  print(pr_stats(opts.run, opts.image_dir, opts.label_db, opts.connected_components_threshold))
