#!/usr/bin/env python3

from PIL import Image
import argparse
import os
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, required=True, help='directory to process')
parser.add_argument('--h', type=int, default=1024, help='target height')
parser.add_argument('--w', type=int, default=768, help='target width')
opts = parser.parse_args()

for fname in os.listdir(opts.dir):
  fname = opts.dir + "/" + fname
  Image.open(fname).resize((opts.w, opts.h), Image.LANCZOS).save(fname)
