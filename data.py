#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageEnhance
from functools import partial
from label_db import LabelDB
import json
import numpy as np
import os
import random
import tensorflow as tf
import util as u

def img_xys_iterator(image_dir, label_dir, batch_size, patch_fraction, distort_rgb,
                     flip_left_right, random_rotation, repeat):
  # return dataset of (image, xys_bitmap) for training

  # materialise list of rgb filenames and corresponding numpy bitmaps
  # (lazy load actual image from filenames as required, but with caching)
  rgb_filenames = []     # (H, W, 3) jpgs
  bitmap_filenames = []  # (H/2, W/2, 1) pngs

  fnames = os.listdir(image_dir)
  random.shuffle(fnames)
  for fname in fnames:
    rgb_filenames.append("%s/%s" % (image_dir, fname))
    bitmap_filenames.append("%s/%s" % (label_dir, fname.replace(".jpg", ".png")))

  def decode_images(rgb_filename, bitmap_filename, w=768, h=1024):
    rgb = tf.image.decode_image(tf.read_file(rgb_filename))
    rgb = tf.reshape(rgb, (h, w, 3))
    rgb = tf.cast(rgb, tf.float32)
    rgb = (rgb / 127.5) - 1.0  # -1.0 -> 1.0
    bitmap = tf.image.decode_image(tf.read_file(bitmap_filename))
    bitmap = tf.reshape(bitmap, (h//2, w//2, 1))
    bitmap = tf.cast(bitmap, tf.float32)
    bitmap /= 256  # 0 -> 1
    return rgb, bitmap

  def random_flip_left_right(rgb, bitmap):
    random = tf.random_uniform([], 0, 1, dtype=tf.float32)
    return tf.cond(random < 0.5,
                   lambda: (rgb, bitmap),
                   lambda: (tf.image.flip_left_right(rgb),
                            tf.image.flip_left_right(bitmap)))

  def random_crop(rgb, bitmap, w=768, h=1024):
    # we want to use the same crop for both RGB input and bitmap labels
    pw, ph = w // patch_fraction, h // patch_fraction
    offset_height = tf.random_uniform([], 0, h-ph, dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, w-pw, dtype=tf.int32)
    rgb = tf.image.crop_to_bounding_box(rgb, offset_height, offset_width, ph, pw)
    bitmap = tf.image.crop_to_bounding_box(bitmap, offset_height // 2, offset_width // 2, ph // 2, pw // 2 )
    return rgb, bitmap

  def distort(rgb, bitmap):
    rgb = tf.image.random_brightness(rgb, 0.1)
    rgb = tf.image.random_contrast(rgb, 0.9, 1.1)
#    rgb = tf.image.per_image_standardization(rgb)  # works great, but how to have it done for predict?
    rgb = tf.clip_by_value(rgb, clip_value_min=-1.0, clip_value_max=1.0)
    return rgb, bitmap

  def rotate(rgb, bitmap):
    # we want to use the same crop for both RGB input and bitmap labels
    random_rotation_angle = tf.random_uniform([], -0.4, 0.4, dtype=tf.float32)
    return (tf.contrib.image.rotate(rgb, random_rotation_angle, 'BILINEAR'),
            tf.contrib.image.rotate(bitmap, random_rotation_angle, 'BILINEAR'))

  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(rgb_filenames),
                                                tf.constant(bitmap_filenames)))
  dataset = dataset.map(decode_images, num_parallel_calls=8)
  if repeat:
    # NOTE: can't cache for very large set (i.e. semi supervised set)
    dataset = dataset.cache().shuffle(100).repeat()
#    dataset = dataset.shuffle(100).repeat()
  if patch_fraction > 1:
    dataset = dataset.map(random_crop, num_parallel_calls=8)
  if flip_left_right:
    dataset = dataset.map(random_flip_left_right, num_parallel_calls=8)
  if distort_rgb:
    dataset = dataset.map(distort, num_parallel_calls=8)
  if random_rotation:
    dataset = dataset.map(rotate, num_parallel_calls=8)
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
  return dataset.batch(1).prefetch(2).make_one_shot_iterator().get_next()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image-dir', type=str, default='images/201802_sample/training',
                      help='location of RGB input images')
  parser.add_argument('--label-dir', type=str, default='labels/201802_sample',
                      help='location of corresponding L label files')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--patch-fraction', type=int, default=1,
                      help="what fraction of image to use as patch. 1 => no patch")
  parser.add_argument('--distort', action='store_true')
  opts = parser.parse_args()
  print(opts)

  from PIL import Image, ImageDraw

  sess = tf.Session()

  imgs, xyss = img_xys_iterator(image_dir=opts.image_dir,
                                label_dir=opts.label_dir,
                                batch_size=opts.batch_size,
                                patch_fraction=opts.patch_fraction,
                                distort_rgb=opts.distort,
                                flip_left_right=True,
                                random_rotation=True,
                                repeat=True)

  for b in range(3):
    print(">batch", b)
    img_batch, xys_batch = sess.run([imgs, xyss])
    for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
      print(">element", i)
      u.side_by_side(rgb=img, bitmap=xys).save("test_%03d_%03d.png" % (b, i))
