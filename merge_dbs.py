#!/usr/bin/env python
from __future__ import print_function

from label_db import LabelDB
import argparse

# merge two dbs into one
# super clumsy :/

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src1', type=str, required=True)
parser.add_argument('--src2', type=str, required=True)
parser.add_argument('--dest', type=str, required=True)
opts = parser.parse_args()

assert opts.src1 != opts.src2

src1 = LabelDB(label_db_file=opts.src1)
imgs1 = src1.imgs()
print("imgs1", len(imgs1))
assert len(imgs1) > 0

src2 = LabelDB(label_db_file=opts.src2)
imgs2 = src2.imgs()
print("imgs2", len(imgs2))
assert len(imgs2) > 0

assert len(imgs1.intersection(imgs2)) == 0

dest = LabelDB(label_db_file=opts.dest)
dest.create_if_required()

for img in imgs1:
  dest.set_labels(img, src1.get_labels(img))
for img in imgs2:
  dest.set_labels(img, src2.get_labels(img))
print("dest", len(dest.imgs()))
