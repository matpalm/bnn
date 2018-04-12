#!/usr/bin/env python3

# go through all sub directories of specified directory and dump
# per day counts

import sys
import os
from collections import Counter, defaultdict

hrs = "05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22".split(" ")
sys.stdout.write("%020s %08s" % ("*dir*", "date"))
for hr in hrs:
  sys.stdout.write(" %4s" % hr)
sys.stdout.write(" %6s\n" % "total")

base_dir = sys.argv[1]
for sub_dir in sorted(os.listdir(base_dir)):
  date_hour_freq = defaultdict(Counter)  # {"20180204": {"08": 3, ...}}
  for filename in os.listdir(base_dir+"/"+sub_dir):
    date, hour = filename[:8], filename[9:11]
    _ = int(date)  # sanity
    _ = int(hour)  # sanity
    date_hour_freq[date][hour] += 1
  if len(date_hour_freq) == 0:
    sys.stdout.write("%020s -\n" % sub_dir)
  else:
    for date in sorted(date_hour_freq.keys()):
      sys.stdout.write("%020s %05s" % (sub_dir, date))
      total_freq = 0
      for hour in hrs:
        freq = date_hour_freq[date][hour]
        total_freq += freq
        if freq == 0:
          sys.stdout.write(" %4s" % "")
        else:
          sys.stdout.write(" %4d" % date_hour_freq[date][hour])
      sys.stdout.write(" %6d\n" % total_freq)
