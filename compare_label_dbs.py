#!/usr/bin/env python3

import argparse
from label_db import LabelDB
import numpy as np
import tensorflow as tf
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--db-1', type=str, required=True, help='db 1. expect every entry is also in db2')
parser.add_argument('--db-2', type=str, required=True, help='db 2. expect to contain every entry of db1')
opts = parser.parse_args()
assert opts.db_1 != opts.db_2

db_1 = LabelDB(label_db_file=opts.db_1)
db_2 = LabelDB(label_db_file=opts.db_2)

# iterate over db_1; every entry must be in db_2
print("\t".join(["img", "#1_total", "#2_total", "ad", "#1_left", "#2_left"]))
for img in db_1.imgs():
  if not db_2.has_labels(img):
    # note: this can imply 0 labels
    raise Exception("img %s is in db_1 but not db_2")
  labels_1 = db_1.get_labels(img)
  labels_2 = db_2.get_labels(img)
  avg_dist, n1_left, n2_left = u.compare_sets(labels_1, labels_2)
  print("\t".join(map(str, [img, len(labels_1), len(labels_2), avg_dist, n1_left, n2_left])))
