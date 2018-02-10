#!/usr/bin/env python3

# go through all sub directories of specified directory and dump
# per day counts

import sys
import os
from collections import Counter

base_dir = sys.argv[1]
for sub_dir in sorted(os.listdir(base_dir)):
  freq = Counter()
  for filename in os.listdir(base_dir+"/"+sub_dir):
    dts = filename[:8]
    _ = int(dts)  # sanity
    freq[dts] += 1
  if len(freq) == 0:
    print(sub_dir, "-", "-")
  else:
    for dts in sorted(freq.keys()):
      print(sub_dir, dts, freq[dts])

