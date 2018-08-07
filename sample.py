#!/usr/bin/env python3

import argparse
import os
import random
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--copy', action='store_true', help='whether to move (dft) or copy file')
parser.add_argument('-n', '--n', type=int, default=10, help='number to sample')
parser.add_argument('--prefix', type=str, default=None, help='prefix to sample from')
parser.add_argument('directories', metavar='directory', type=str, nargs=2, help='src_dir dest_dir')
opts = parser.parse_args()
print("# opts %s" % opts, file=sys.stderr)

if opts.n <= 0:
  raise Exception("can only sample +ve number, not %d" % opts.n)

src_directory, dest_directory = opts.directories
if not os.path.isdir(src_directory):
  raise Exception("src_directory [%s] doesn't exist" % src_directory)
if not os.path.isdir(dest_directory):
  print("creating destination directory [%s]" % dest_directory, file=sys.stderr)
  print("mkdir -p \"%s\"" % dest_directory)

src_files = os.listdir(src_directory)
print("%d src_files from [%s]" % (len(src_files), src_directory), file=sys.stderr)

if opts.prefix is not None:
  src_files = list(filter(lambda f: f.startswith(opts.prefix), src_files))
  print("after filtering with prefix [%s] we have %d files" % (opts.prefix, len(src_files)), file=sys.stderr)

if len(src_files) < opts.n:
  raise Exception("requested sample of %d files from directory [%s] but there are only %d files there" %
                  (opts.n, src_directory, len(src_files)))

random.shuffle(src_files)
for filename in src_files[:opts.n]:
  print("%s \"%s/%s\" \"%s\"" % ("cp" if opts.copy else "mv",
                         src_directory, filename, dest_directory))
