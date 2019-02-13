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
                     flip_left_right, random_rotation, repeat, width, height,
                     label_rescale=0.5):
  # return dataset of (image, xys_bitmap) for training

  # give super explicit exception re: setting patch_width_height and width, height
  width_height_set = False
  if width is not None or height is not None:
    if width is None or height is None:
      raise Exception("when setting --width or --height must set both")
    width_height_set = True
  patch_set = patch_width_height is not None
  if patch_set and width_height_set:
    raise Exception("need to set either --patch-width-height or --width and --height, not both")
  if not (patch_set or width_height_set):
    raise Exception("need to set one of either --patch-width-height or both --width and --height")

  # materialise list of rgb filenames and corresponding numpy bitmaps
  rgb_filenames = []     # (H, W, 3) jpgs
  bitmap_filenames = []  # (H/2, W/2, 1) pngs
  for fname in os.listdir(image_dir):
    rgb_filename = "%s/%s" % (image_dir, fname)
    rgb_filenames.append(rgb_filename)
    bitmap_filename = "%s/%s" % (label_dir, fname.replace(".jpg", ".png"))
    if not os.path.isfile(bitmap_filename):
      raise Exception("label bitmap img [%s] doesn't exist for training example [%s]. did you run materialise_label_db.py?"
                      % (bitmap_filename, rgb_filename))
    bitmap_filenames.append(bitmap_filename)

  def decode_images(rgb_filename, bitmap_filename):
    rgb = tf.image.decode_image(tf.read_file(rgb_filename))
    rgb = tf.cast(rgb, tf.float32)
    rgb = (rgb / 127.5) - 1.0  # -1.0 -> 1.0
    bitmap = tf.image.decode_image(tf.read_file(bitmap_filename))
    bitmap = tf.cast(bitmap, tf.float32)
    bitmap /= 256  # 0 -> 1
    return rgb, bitmap

  def random_crop(rgb, bitmap):
    # we want to use the same crop for both RGB input and bitmap labels
    patch_width = patch_height = patch_width_height
    height, width = tf.shape(rgb)[0], tf.shape(rgb)[1]
    offset_height = tf.random_uniform([], 0, height-patch_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], 0, width-patch_width, dtype=tf.int32)
    rgb = tf.image.crop_to_bounding_box(rgb, offset_height, offset_width, patch_height, patch_width)
    rgb = tf.reshape(rgb, (patch_height, patch_width, 3))
    # TODO: remove this cast uglyness :/
    bitmap = tf.image.crop_to_bounding_box(bitmap,
                                           tf.cast(tf.cast(offset_height, tf.float32) * label_rescale, tf.int32),
                                           tf.cast(tf.cast(offset_width, tf.float32) * label_rescale, tf.int32),
                                           int(patch_height * label_rescale), int(patch_width * label_rescale))
    bitmap = tf.reshape(bitmap, (int(patch_height * label_rescale), int(patch_width * label_rescale), 1))
    return rgb, bitmap

  def augment(rgb, bitmap):
    if flip_left_right:
      random = tf.random_uniform([], 0, 1, dtype=tf.float32)
      rgb, bitmap = tf.cond(random < 0.5,
                            lambda: (rgb, bitmap),
                            lambda: (tf.image.flip_left_right(rgb),
                                     tf.image.flip_left_right(bitmap)))
    if distort_rgb:
      rgb = tf.image.random_brightness(rgb, 0.1)
      rgb = tf.image.random_contrast(rgb, 0.9, 1.1)
      #    rgb = tf.image.per_image_standardization(rgb)  # works great, but how to have it done for predict?
      rgb = tf.clip_by_value(rgb, clip_value_min=-1.0, clip_value_max=1.0)

    if random_rotation:
      # we want to use the same crop for both RGB input and bitmap labels
      random_rotation_angle = tf.random_uniform([], -0.4, 0.4, dtype=tf.float32)
      rgb, bitmap = (tf.contrib.image.rotate(rgb, random_rotation_angle, 'BILINEAR'),
                     tf.contrib.image.rotate(bitmap, random_rotation_angle, 'BILINEAR'))

    return rgb, bitmap

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

  if flip_left_right or distort_rgb or random_rotation:
    dataset = dataset.map(augment, num_parallel_calls=8)

  # NOTE: keras.fit wants the iterator directly (not .get_next())
  return dataset.batch(batch_size).prefetch(tf.contrib.data.AUTOTUNE)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image-dir', type=str, default='sample_data/training/',
                      help='location of RGB input images')
  parser.add_argument('--label-dir', type=str, default='sample_data/labels/',
                      help='location of corresponding L label files. (note: we assume for'
                           'each image-dir image there is a label-dir image)')
  parser.add_argument('--batch-size', type=int, default=4)
  parser.add_argument('--patch-width-height', type=int, default=None,
                      help="what size square patches to sample. None => no patch, i.e. use full res image"
                           " (in which case --width & --height are required)")
  parser.add_argument('--label-rescale', type=float, default=0.5,
                      help='relative scale of label bitmap compared to input image')
  parser.add_argument('--distort', action='store_true')
  parser.add_argument('--rotate', action='store_true')
  parser.add_argument('--width', type=int, default=768,
                      help='input image width. required if --patch-width-height not set.')
  parser.add_argument('--height', type=int, default=1024,
                      help='input image height. required if --patch-width-height not set.')
  opts = parser.parse_args()
  print(opts)

  from PIL import Image, ImageDraw

  sess = tf.Session()

  imgs_xyss = img_xys_iterator(image_dir=opts.image_dir,
                               label_dir=opts.label_dir,
                               batch_size=opts.batch_size,
                               patch_width_height=opts.patch_width_height,
                               distort_rgb=opts.distort,
                               flip_left_right=opts.distort,
                               random_rotation=opts.rotate,
                               repeat=True,
                               width=None if opts.patch_width_height else opts.width,
                               height=None if opts.patch_width_height else opts.height,
                               label_rescale=opts.label_rescale)

  imgs, xyss = imgs_xyss.make_one_shot_iterator().get_next()

  for b in range(3):
    img_batch, xys_batch = sess.run([imgs, xyss])
    for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
      fname = "test_%03d_%03d.png" % (b, i)
      print("batch", b, "element", i, "fname", fname)
      u.side_by_side(rgb=img, bitmap=xys).save(fname)
