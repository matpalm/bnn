#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageEnhance
from functools import partial
import json
import os
import random
import tensorflow as tf
import numpy as np
from label_db import LabelDB

def img_xys_iterator(base_dir, batch_size, patch_fraction, distort):
  # load all images
  label_db = LabelDB(check_same_thread=False)
  rgb_images = []  
  label_bitmaps = []
  w, h = None, None
  for fname in os.listdir(base_dir):
    # load image
    img = Image.open("%s/%s" % (base_dir, fname))
    w, h = img.size
    # lookup xys
    xys = label_db.get_labels(fname)
    # keep track for later stacking
    rgb_images.append(np.array(img))
    label_bitmaps.append(xys_to_bitmap(xys, h, w, rescale=0.5))

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

  dataset = tf.data.Dataset.from_tensor_slices((np.stack(rgb_images),
                                                np.stack(label_bitmaps)))
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
      
def xys_to_bitmap(xys, height, width, rescale=1.0):
  # note: include trailing 1 dim to easier match model output
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  for x, y in xys:
    bitmap[int(y*rescale), int(x*rescale), 0] = 1.0  # recall images are (height, width)
  return bitmap

# TODO: move to util.py
def bitmap_to_pil_image(bitmap):
  h, w, c = bitmap.shape
  assert c == 1
  rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
  single_channel = bitmap[:,:,0] * 255
  rgb_array[:,:,0] = single_channel
  rgb_array[:,:,1] = single_channel
  rgb_array[:,:,2] = single_channel
  return Image.fromarray(rgb_array)
  

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
                                distort=opts.distort)

  for b in range(3):
    print(">batch", b)
    img_batch, xys_batch = sess.run([imgs, xyss])
    for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
      print(">element", i)
      h, w, _ = img.shape    
      canvas = Image.new('RGB', (w*2, h), (50, 50, 50))
      # paste RGB on left hand side
      img = Image.fromarray(img)
      canvas.paste(img, (0, 0))
      # paste bitmap version of labels on right hand side
      # black with white dots at labels
      img = bitmap_to_pil_image(xys)
      img = img.resize((w, h))
      canvas.paste(img, (w, 0))
      # draw on a blue border (and blue middle divider) to make it
      # easier to see relative positions.
      draw = ImageDraw.Draw(canvas)
      draw.polygon([0,0,w*2-1,0,w*2-1,h-1,0,h-1], outline='blue')
      draw.line([w,0,w,h], fill='blue')
      # save
      canvas.save("test_%03d_%03d.png" % (b, i))
    
    


