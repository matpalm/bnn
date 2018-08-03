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
set_comparison = u.SetComparison()
for img in predicted_db.imgs():
  if not true_db.has_labels(img):
    # note: this can imply 0 labels
    raise Exception("img %s is in --predicted-db but not --true-db")
  true_labels = true_db.get_labels(img)
  predicted_labels = predicted_db.get_labels(img)
  TP, FP, FN = set_comparison.compare_sets(true_labels, predicted_labels)
  print("img", img, TP, FP, FN)

# dump overall summary
print("TP", set_comparison.true_positive_count,
      "FP", set_comparison.false_positive_count,
      "FN", set_comparison.false_negative_count)
print("precision %0.3f  recall %0.3f  f1 %0.3f" % set_comparison.precision_recall_f1())
