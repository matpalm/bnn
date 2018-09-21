#!/usr/bin/env python3

from label_db import LabelDB
import argparse

# merge two dbs into one
# super clumsy :/

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--from-db', type=str, required=True, help='db to take entries from')
parser.add_argument('--into-db', type=str, required=True, help='db to add entries to')
opts = parser.parse_args()

assert opts.from_db != opts.into_db

from_db = LabelDB(label_db_file=opts.from_db)
if not from_db.has_been_created():
  raise Exception("--from-db has never been populated")

into_db = LabelDB(label_db_file=opts.into_db)
if not into_db.has_been_created():
  raise Exception("--into-db has never been populated")

num_ignored = 0
num_added = 0
for img in from_db.imgs():
  if into_db.has_labels(img):
    print("ignore", img, "; already in into_db")
    num_ignored += 1
  else:
    into_db.set_labels(img, from_db.get_labels(img))
    num_added += 1

print("num_ignored", num_ignored, "num_added", num_added)
