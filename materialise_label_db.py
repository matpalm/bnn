#!/usr/bin/env python3

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
opts = parser.parse_args()
print(opts)

label_db = LabelDB(label_db_file=opts.label_db)

if not os.path.exists(opts.directory):
  os.makedirs(opts.directory)

fnames = list(label_db.imgs())
for i, fname in enumerate(fnames):
  bitmap = u.xys_to_bitmap(xys=label_db.get_labels(fname),
                           height=1024,
                           width=768,
                           rescale=0.5)
  single_channel_img = u.bitmap_to_single_channel_pil_image(bitmap)
  single_channel_img.save("%s/%s" % (opts.directory, fname.replace(".jpg", ".png")))
  sys.stdout.write("%d/%d   \r" % (i, len(fnames)))
