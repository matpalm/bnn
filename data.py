#!/usr/bin/env python
from __future__ import print_function

from PIL import Image, ImageDraw, ImageEnhance
from functools import partial
from label_db import LabelDB
import json
import numpy as np
import os
import random
import tensorflow as tf
import util as u

def img_xys_iterator(base_dir, batch_size, patch_fraction, distort, repeat):
  # return dataset of (image, xys_bitmap) for training

  label_db = LabelDB(check_same_thread=False)

  # materialise list of filenames and all labels
  # (lazy load actual image from filename as required, but with caching)
  filenames = []
  label_bitmaps = []
  w, h = 768, 1024
  for fname in sorted(os.listdir(base_dir)):
    # keep track of filename
    filenames.append("%s/%s" % (base_dir, fname))
    # keep track of labels bitmap
    label_bitmaps.append(xys_to_bitmap(xys=label_db.get_labels(fname),
                                       height=h,
                                       width=w,
                                       rescale=0.5))

  def decode_image(filename, bitmap):
    image = tf.image.decode_image(tf.read_file(filename))
    image = tf.reshape(image, (h, w, 3))  # only required for debugging?
    return image, bitmap
  
  def random_crop(rgb, bitmap):
    # we want to use the same crop for both RGB input and bitmap label
    # but the bitmap is half res
    pw, ph = w / patch_fraction, h / patch_fraction
    offset_height = tf.random_uniform([], 0, h-ph, dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, w-pw, dtype=tf.int32)
    rgb = tf.image.crop_to_bounding_box(rgb, offset_height, offset_width, ph, pw)
    bitmap = tf.image.crop_to_bounding_box(bitmap, offset_height/2, offset_width/2, ph/2, pw/2)
    return rgb, bitmap

  def distort_rgb(rgb, bitmap):
    rgb = tf.image.random_brightness(rgb, 0.1)
    rgb = tf.image.random_contrast(rgb, 0.9, 1.1)
    # TODO: work out how best to introduce this (since it's probably a good thing!)
    #       without breaking everything to do with debugging images
#    rgb = tf.image.per_image_standardization(rgb)
    return rgb, bitmap

  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames),
                                                np.stack(label_bitmaps)))  
  dataset = dataset.map(decode_image, num_parallel_calls=8)  
  if repeat:
    dataset = dataset.cache().shuffle(50).repeat()
  if patch_fraction > 1:
    dataset = dataset.map(random_crop, num_parallel_calls=8)
  if distort:
    dataset = dataset.map(distort_rgb, num_parallel_calls=8)
  return (dataset.
          batch(batch_size).
          prefetch(1).
          make_one_shot_iterator().
          get_next())

def img_filename_iterator(base_dir):
  # return dataset of (image, filename) for inference
  
  # decide both local path filenames and full path for decoding
  filenames = sorted(os.listdir(base_dir))
  full_path_filenames = [ "%s/%s" % (base_dir, f) for f in filenames ]

  def decode_image(full_path_filename, filename):
    image = tf.image.decode_image(tf.read_file(full_path_filename))
    image = tf.reshape(image, (1024, 768, 3))  # only required for debugging?
    return image, filename  # reemit filename as "label"

  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(full_path_filenames),
                                                tf.constant(filenames)))
  dataset = dataset.map(decode_image, num_parallel_calls=8)
  return dataset.batch(1).prefetch(1).make_one_shot_iterator().get_next()

def xys_to_bitmap(xys, height, width, rescale=1.0):
  # note: include trailing 1 dim to easier match model output
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  for x, y in xys:
    bitmap[int(y*rescale), int(x*rescale), 0] = 1.0  # recall images are (height, width)
  return bitmap
  
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image-dir', type=str, default="images/sample_originals/train")
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--patch-fraction', type=int, default=1,
                      help="what fraction of image to use as patch. 1 => no patch")
  parser.add_argument('--distort', action='store_true')
  opts = parser.parse_args()
  
  from PIL import Image, ImageDraw

  sess = tf.Session()
  
  imgs, xyss = img_xys_iterator(base_dir=opts.image_dir,
                                batch_size=opts.batch_size,
                                patch_fraction=opts.patch_fraction,
                                distort=opts.distort,
                                repeat=True)

  for b in range(3):
    print(">batch", b)
    img_batch, xys_batch = sess.run([imgs, xyss])
    for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
      print(">element", i)
      u.side_by_side(rgb=img, bitmap=xys).save("test_%03d_%03d.png" % (b, i))
    
