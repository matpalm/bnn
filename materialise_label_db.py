#!/usr/bin/env python3

# given a label_db create a single channel image corresponding to each image.

from PIL import Image
from label_db import LabelDB
import argparse
import numpy as np
import os
import sys
import util as u

# TODO: make this multiprocess, too slow as is...

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label-db', type=str, help='label_db to materialise bitmaps from')
parser.add_argument('--directory', type=str, help='directory to store bitmaps')
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
parser.add_argument('--label-rescale', type=float, default=0.5,
                    help='relative scale of label bitmap compared to input image')
opts = parser.parse_args()
print(opts)

label_db = LabelDB(label_db_file=opts.label_db)

if not os.path.exists(opts.directory):
  os.makedirs(opts.directory)

fnames = list(label_db.imgs())
for i, fname in enumerate(fnames):
  bitmap = u.xys_to_bitmap(xys=label_db.get_labels(fname),
                           height=opts.height,
                           width=opts.width,
                           rescale=opts.label_rescale)
  single_channel_img = u.bitmap_to_single_channel_pil_image(bitmap)
  single_channel_img.save("%s/%s" % (opts.directory, fname.replace(".jpg", ".png")))
  sys.stdout.write("%d/%d   \r" % (i, len(fnames)))
