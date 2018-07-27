#!/usr/bin/env python3
from PIL import Image, ImageDraw
import label_db
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
opts = parser.parse_args()

HW = 50//2  # crops are 50x50

def valid(labels, cx, cy):
  # is idxth item in labels "valid"? where "valid" => bounding
  # box within image and no other bee in bounding box.
  x1, y1 = cx-HW, cy-HW
  if x1 < 0 or y1 < 0:
    # either left or top margin out of bounds => invalid
    return False
  x2, y2 = cx+HW, cy+HW
  if x2 > opts.width or y2 > opts.height:
    # either right or bottom margin out of bounds => invalid
    return False
  for ox, oy in labels:
    if ox == cx and oy == cy:
      # this 'other' bee is the one being checked => ignore
      continue
    if x1 < ox and ox < x2 and y1 < oy and oy < y2:
      # other bee inside bounding box => invalid
      return False
  return True

#canvas = Image.new('RGB', (HW*20, HW*20), (0,0,0))
db = label_db.LabelDB(label_db_file='label.201802_sample.db')
for img_fname in list(db.imgs()):
  out_idx = 0
  img = None
  labels = db.get_labels(img_fname)
  for cx, cy in labels:
    print("check", img_fname, cx, cy)
    if not valid(labels, cx, cy):
      continue
    if img is None:
      img_dname = img_fname[:8]
      img = Image.open("images/%s/%s" % (img_dname, img_fname))
    crop = img.crop((cx-HW, cy-HW, cx+HW, cy+HW))
#    px, py = (i%20)*HW, (i//20)*HW
#    print("!", i, px, py)
#    canvas.paste(crop, (px, py))
    crop.save("images/single_bees/%s_%03d.png" % (img_fname.replace(".jpg", ""), out_idx))
    out_idx += 1
#    if i > (20*20):
#      canvas.save("foo.png")
#      exit()
