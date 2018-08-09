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

def img_xys_iterator(image_dir, label_dir, batch_size, patch_width_height, distort_rgb,
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

  # check images are valid and compute width and height
  width, height = u.check_images(rgb_filenames)

  def decode_images(rgb_filename, bitmap_filename):
    rgb = tf.image.decode_image(tf.read_file(rgb_filename))
    rgb = tf.cast(rgb, tf.float32)
    rgb = (rgb / 127.5) - 1.0  # -1.0 -> 1.0
    bitmap = tf.image.decode_image(tf.read_file(bitmap_filename))
    bitmap = tf.cast(bitmap, tf.float32)
    bitmap /= 256  # 0 -> 1
    return rgb, bitmap

  def random_flip_left_right(rgb, bitmap):
    random = tf.random_uniform([], 0, 1, dtype=tf.float32)
    return tf.cond(random < 0.5,
                   lambda: (rgb, bitmap),
                   lambda: (tf.image.flip_left_right(rgb),
                            tf.image.flip_left_right(bitmap)))

  def random_crop(rgb, bitmap):
    # we want to use the same crop for both RGB input and bitmap labels
    patch_width = patch_height = patch_width_height
    height, width = tf.shape(rgb)[0], tf.shape(rgb)[1]
    offset_height = tf.random_uniform([], 0, height-patch_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, width-patch_width, dtype=tf.int32)
    rgb = tf.image.crop_to_bounding_box(rgb, offset_height, offset_width, patch_height, patch_width)
    rgb = tf.reshape(rgb, (patch_height, patch_width, 3))
    bitmap = tf.image.crop_to_bounding_box(bitmap, offset_height // 2, offset_width // 2,
                                           patch_height // 2, patch_width // 2 )
    bitmap = tf.reshape(bitmap, (patch_height // 2, patch_width // 2, 1))
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

  def set_explicit_size(rgb, bitmap):
    if height is None or width is None:
      raise Exception(">set_explicit_size requires explicit height/width set when not patch sampling")
    return (tf.reshape(rgb, (height, width, 3)),
            tf.reshape(bitmap, (height // 2, width // 2, 1)))

  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(rgb_filenames),
                                                tf.constant(bitmap_filenames)))

  dataset = dataset.map(decode_images, num_parallel_calls=8)

  if repeat:
    if len(rgb_filenames) < 1000:
      dataset = dataset.cache()
    print("len(rgb_filenames)", len(rgb_filenames), ("CACHE" if len(rgb_filenames) < 1000 else "NO CACHE"))
    dataset = dataset.shuffle(1000).repeat()

  if patch_width_height is not None:
    dataset = dataset.map(random_crop, num_parallel_calls=8)
  else:
    # this is clumsy but required for rotation as well as current debugging.
    # TODO: refactor away from requiring this (and, by implication, --height, --width)
    dataset = dataset.map(set_explicit_size, num_parallel_calls=8)

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


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i', '--image-dir', type=str, default='sample_data/training/',
                      help='location of RGB input images')
  parser.add_argument('-l', '--label-dir', type=str, default='sample_data/labels/',
                      help='location of corresponding L label files. (note: we assume for'
                           'each image-dir image there is a label-dir image)')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--patch-width-height', type=int, default=None,
                      help="what size square patches to sample. None => no patch, i.e. use full res image"
                           " (in which case --width & --height are required)")
  parser.add_argument('--distort', action='store_true')
  parser.add_argument('--rotate', action='store_true')
  opts = parser.parse_args()
  print(opts)

  from PIL import Image, ImageDraw

  sess = tf.Session()

  imgs, xyss = img_xys_iterator(image_dir=opts.image_dir,
                                label_dir=opts.label_dir,
                                batch_size=opts.batch_size,
                                patch_width_height=opts.patch_width_height,
                                distort_rgb=opts.distort,
                                flip_left_right=True,
                                random_rotation=opts.rotate,
                                repeat=True)

  for b in range(3):
    img_batch, xys_batch = sess.run([imgs, xyss])
    for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
      fname = "test_%03d_%03d.png" % (b, i)
      print("batch", b, "element", i, "fname", fname)
      u.side_by_side(rgb=img, bitmap=xys).save(fname)
