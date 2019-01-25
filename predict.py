#!/usr/bin/env python3

# given a directory of images output a list of image -> predictions

from PIL import Image, ImageDraw
from label_db import LabelDB
import argparse
import itertools
#import kmodel
import numpy as np
from tensorflow.keras.models import load_model
import os
import random
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--num', type=int, default=None,
                    help='if set run prediction for this many random images. if not set run for all')
parser.add_argument('--output-label-db', type=str, default=None, help='if not set dont write label_db')
parser.add_argument('--run', type=str, required=True, help='model, also used as subdir for export-pngs')
#parser.add_argument('--no-use-skip-connections', action='store_true')
#parser.add_argument('--no-use-batch-norm', action='store_true')
parser.add_argument('--export-pngs', default='',
                    help='how, if at all, to export pngs {"", "predictions", "centroids"}')
#parser.add_argument('--base-filter-size', type=int, default=8)
#parser.add_argument('--connected-components-threshold', type=float, default=0.05)
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
opts = parser.parse_args()



latest_model = "ckpts/%s/%s" % (opts.run, u.last_file_in_dir("ckpts/%s" % opts.run))
print("using model [%s]" % latest_model)
model = load_model(latest_model)
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
  print("img", img.shape)

  # run through model (adding / removing dummy batch)
  # TODO: do this in batch !!
  prediction = model.predict(np.expand_dims(img, 0))[0]
  print("prediction", prediction.shape)

  # calc [(x,y), ...] centroids
#  centroids = u.centroids_of_connected_components(prediction,
#                                                  rescale=2.0,
#                                                  threshold=opts.connected_components_threshold)

#  print("\t".join(map(str, [idx, filename, len(centroids)])))

  # export some debug image (if requested)
  if opts.export_pngs != '':
    if opts.export_pngs == 'predictions':
      debug_img = u.side_by_side(rgb=img, bitmap=prediction)
    elif opts.export_pngs == 'centroids':
      debug_img = u.red_dots(rgb=img, centroids=centroids)
    else:
      raise Exception("unknown --export-pngs option")
    print("save debug_img %s/%s.png" % (export_dir, filename))
    debug_img.save("%s/%s.png" % (export_dir, filename))

  # set new labels (if requested)
  if db:
    db.set_labels(filename, centroids, flip=True)
