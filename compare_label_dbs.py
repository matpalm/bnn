#!/usr/bin/env python3

import argparse
from label_db import LabelDB
import numpy as np
import util as u

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--true-db', type=str, required=True, help='true labels')
parser.add_argument('--predicted-db', type=str, required=True, help='predicted labels')
opts = parser.parse_args()
assert opts.true_db != opts.predicted_db

true_db = LabelDB(label_db_file=opts.true_db)
predicted_db = LabelDB(label_db_file=opts.predicted_db)

# iterate over predicted_db; we expect true_db to be a super set of it.
print("\t".join(["img", "#1_total", "#2_total", "ad", "#1_left", "#2_left"]))
total_TP = total_FP = total_FN = 0
for img in predicted_db.imgs():
  if not true_db.has_labels(img):
    # note: this can imply 0 labels
    raise Exception("img %s is in --predicted-db but not --true-db")

  true_labels = true_db.get_labels(img)
  predicted_labels = predicted_db.get_labels(img)
  TP, FP, FN = u.compare_sets(true_labels, predicted_labels)
  print("img", img, TP, FP, FN)

  total_TP += TP
  total_FP += FP
  total_FN += FN

precision = total_TP / ( total_TP + total_FP )
recall = total_TP / ( total_TP + total_FN )
f1 = 2 * (precision * recall) / (precision + recall)
print("precision %0.3f  recall %0.3f  f1 %0.3f" % ( precision, recall, f1))
