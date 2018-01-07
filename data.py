#!/usr/bin/env python
from PIL import Image
from functools import partial
import json
import os
import random
import tensorflow as tf
import numpy as np
from label_db import LabelDB

def img_xys_iterator(base_dir, batch_size, img_shape):
  dataset = tf.data.Dataset.from_generator(partial(_decode_img_and_lookup_xys,
                                                   base_dir,
                                                   for_training=False,
                                                   img_shape=img_shape),
                                           output_types=(tf.uint8, tf.uint8))
  dataset = dataset.batch(batch_size).prefetch(2)
  return dataset.make_one_shot_iterator().get_next()

def _decode_img_and_lookup_xys(base_dir, for_training, img_shape):
  label_db = LabelDB(check_same_thread=False)
  while True:
    for fname in os.listdir(base_dir):
      
      fullname = "%s/%s" % (base_dir, fname)      
      img = np.array(Image.open(fullname))
      h, w, _ = img.shape

      # TODO: drop use of img_shape completely
      
      # TODO: distort here
  
      # TODO: lookup xys
      xys = label_db.get_labels(fullname)
      print("xys", fullname, xys)
        
#      if img_shape == '256':
#        xys = [(33, 26), (162, 143), (169, 221)]
#      elif img_shape == '48':
#        xys = [(29, 46), (29, 37), (15, 23)]
#      elif img_shape == '384':
#        xys = [(242, 302), (116, 182), (251, 377), (92, 79), (69, 149), (60, 109), (56, 150), (64, 131), (84, 145)]
#      else:
#        raise Exception("??? %s" % img_shape)
  
      # done
      yield img, xys_to_bitmap(xys, h, w, rescale=0.5)

def xys_to_bitmap(xys, height, width, rescale=1.0):
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
#  print "BB", bitmap.shape
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
  parser.add_argument('--base-dir', type=str, default="images/sample_originals")
#  parser.add_argument('--labelled-data-fname', type=str, default="train.jsons")
#  parser.add_argument('--img-shape', type=str, default="384")
  opts = parser.parse_args()
  
  from PIL import Image, ImageDraw

  sess = tf.Session()
  
  imgs, xys = img_xys_iterator(base_dir=opts.base_dir,
                               batch_size=5,
                               img_shape="not_used") #opts.img_shape)
  
  img_batch, xys_batch = sess.run([imgs, xys])
  for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
    h, w, _ = img.shape
    
    canvas = Image.new('RGB', (w*2, h), (0, 0, 0))

    # paste RGB on left hand side
    img = Image.fromarray(img)
    canvas.paste(img, (0, 0))

    # paste bitmap version of labels on right hand side
    # black with white dots at labels
    img = bitmap_to_pil_image(xys)
    img = img.resize((h, w))
    canvas.paste(img, (w, 0))
    
    canvas.save("test_%d.png" % i)
    
    


