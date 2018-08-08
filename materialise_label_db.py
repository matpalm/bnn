#!/usr/bin/env python3

from PIL import Image
from label_db import LabelDB
import argparse
import numpy as np
import os
import sys
import util as u
from shutil import copy

# TODO: make this multiprocess, too slow as is...

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--label-db', type=str, help='label_db to materialise bitmaps from')
parser.add_argument('-i', '--image-dir', required=True, type=str, help='directory to images')
parser.add_argument('-d', '--directory', required=True, type=str, help='directory to store bitmaps')
parser.add_argument('-t', '--training-dir', type=str, help='copy labelled images to training folder')
opts = parser.parse_args()
print(opts)

label_db = LabelDB(label_db_file=opts.label_db)

fnames = list(label_db.imgs())

# verify all the image files are okay and the same size
imgnames = [os.path.join(opts.image_dir, imgname) for imgname in fnames]
width, height = u.check_images(imgnames)

if not os.path.exists(opts.directory):
  os.makedirs(opts.directory)

# train.py expects the images to match the labels so this copies labelled images
if opts.training_dir is not None:
  if not os.path.exists(opts.training_dir):
    os.makedirs(opts.training_dir)
  for img in imgnames:
    copy(img, opts.training_dir)

for i, fname in enumerate(fnames):
  bitmap = u.xys_to_bitmap(xys=label_db.get_labels(fname),
                           height=height, width=width,
                           rescale=0.5)
  single_channel_img = u.bitmap_to_single_channel_pil_image(bitmap)
  single_channel_img.save("%s/%s" % (opts.directory, fname.replace(".jpg", ".png")))
  sys.stdout.write("%d/%d   \r" % (i+1, len(fnames)))