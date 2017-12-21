#!/usr/bin/env python
from PIL import Image
from functools import partial
import json
import os
import random
import tensorflow as tf
import numpy as np

def img_xys_iterator(base_dir, batch_size, img_shape):
  dataset = tf.data.Dataset.from_generator(partial(_fnames,
                                                   base_dir),
                                           output_types=tf.string)  # filename
  dataset = dataset.map(partial(_decode_img_and_lookup_xys,
                                for_training=False,
                                img_shape=img_shape),
                        num_parallel_calls=8)
  dataset = dataset.batch(batch_size).prefetch(2)
  return dataset.make_one_shot_iterator().get_next()

def _fnames(base_dir):
  while True:
    for fname in os.listdir(base_dir):
      yield "%s/%s" % (base_dir, fname)

def _decode_img_and_lookup_xys(fname, for_training, img_shape):
  png_bytes = tf.read_file(fname)
  img = tf.image.decode_image(png_bytes)

  # do explicit reshape (annoyingly) required for a bunch of shape
  # inference, but also used for lable bitmap. TODO: remove this hack!
  
  if img_shape == '256':
    h, w = 256, 256
#  elif img_shape == '512':
#    h, w = 512, 512
  elif img_shape == '384':
    h, w = 512, 384
  elif img_shape == '48':
    h, w = 48, 64
  else:
    raise ValueError("Unexpected img_shape %s" % img_shape)

  img = tf.reshape(img, (h, w, 3))
  # distort image; if training
  # if for_training:
  #   img = tf.contrib.image.rotate(img, tf.random_uniform([], -0.1, 0.1), 'BILINEAR')
  #   img = tf.image.adjust_brightness(img, tf.random_uniform([], -0.1, 0.1))
  #   img = tf.image.adjust_saturation(img, tf.random_uniform([], 0.7, 1.3))
  #   img = tf.image.adjust_contrast(img, tf.random_uniform([], 0.9, 1.1))

  # TODO: lookup xys
#  print "fname!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", fname
  if img_shape == '256':
    xys = [(33, 26), (162, 143), (169, 221)]
  elif img_shape == '48':
    xys = [(29, 46), (29, 37), (15, 23)]
  elif img_shape == '384':
    xys = [(242, 302), (116, 182), (251, 377), (92, 79), (69, 149), (60, 109), (56, 150), (64, 131), (84, 145)]
  else:
    raise Exception("??? %s" % img_shape)
  # done
  return img, xys_to_bitmap(xys, h, w, rescale=0.5)

def xys_to_bitmap(xys, height, width, rescale=1.0):
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  print "BB", bitmap.shape
  for x, y in xys:
    bitmap[int(x*rescale), int(y*rescale), 0] = 1.0
  return bitmap

def bitmap_to_pil_image(bitmap):
  rgb_array = np.zeros((bitmap.shape[1], bitmap.shape[0], 3), dtype=np.uint8)
  single_channel = bitmap[:,:,0].T * 255
  rgb_array[:,:,0] = single_channel
  rgb_array[:,:,1] = single_channel
  rgb_array[:,:,2] = single_channel
  return Image.fromarray(rgb_array)
  

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--base-dir', type=str, default="images/test2")
#  parser.add_argument('--labelled-data-fname', type=str, default="train.jsons")
  parser.add_argument('--img-shape', type=str, default="384")
  opts = parser.parse_args()
  
  from PIL import Image, ImageDraw

  sess = tf.Session()
  
  imgs, xys = img_xys_iterator(base_dir=opts.base_dir,
                               batch_size=1,
                               img_shape=opts.img_shape)
  
  img_batch, xys_batch = sess.run([imgs, xys])
  for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
    h, w, _ = img.shape
    print "img", img.shape, "xys", xys.shape 
    
    canvas = Image.new('RGB', (w*2, h), (0, 0, 0))
    
    img = Image.fromarray(img)
    canvas.paste(img, (0, 0))

    img = bitmap_to_pil_image(xys)
    img = img.resize((h, w))
    canvas.paste(img, (w, 0))
    
    canvas.save("test_%d.png" % i)
    
    


