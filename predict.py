#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
from label_db import LabelDB
from scipy.special import expit
import argparse
import model as m
import numpy as np
import os
import random
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--num', type=int, default=None,
                    help='if set run prediction for this many random images. if not set run for all')
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--run', type=str, required=True, help='model, also used as subdir for export-pngs')
parser.add_argument('--export-pngs', default='',
                    help='how, if at all, to export pngs {"", "predictions", "centroids"}')
opts = parser.parse_args()

train_opts, model =  m.restore_model(opts.run)
print(model.summary())

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

imgs = os.listdir(opts.image_dir)
if opts.num is not None:
  assert opts.num > 0
  imgs = random.sample(imgs, opts.num)

for idx, filename in enumerate(sorted(imgs)):

  # load next image
  img = np.array(Image.open(opts.image_dir+"/"+filename))  # uint8 0->255  (H, W)
  img = img.astype(np.float32)
  img = (img / 127.5) - 1.0  # -1.0 -> 1.0  # see data.py

  # run through model (adding / removing dummy batch)
  # recall: output from model is logits so we need to expit
  # TODO: do this in batch !!
  prediction = expit(model.predict(np.expand_dims(img, 0))[0])

  # calc [(x,y), ...] centroids
  centroids = u.centroids_of_connected_components(prediction,
                                                  rescale=2.0,
                                                  threshold=train_opts['connected_components_threshold'])
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
