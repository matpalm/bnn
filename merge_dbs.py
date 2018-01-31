#!/usr/bin/env python
from __future__ import print_function

from label_db import LabelDB
import argparse

# merge two dbs into one
# super clumsy :/

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--from-db', type=str, required=True, help='db to take entries from')
parser.add_argument('--to-db', type=str, required=True, help='db to add entries to')
opts = parser.parse_args()

assert opts.from_db != opts.to_db

from_db = LabelDB(label_db_file=opts.from_db)
to_db = LabelDB(label_db_file=opts.to_db)
      
num_ignored = 0
num_added = 0
for img in from_db.imgs():
  if to_db.has_labels(img):
    print("ignore", img, "; already in to_db")
    num_ignored += 1    
  else:
    to_db.set_labels(img, from_db.get_labels(img))
    num_added += 1

print("num_ignored", num_ignored, "num_added", num_added)
