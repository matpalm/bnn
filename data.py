#!/usr/bin/env python
from PIL import Image
from functools import partial
import json
import os
import random
import tensorflow as tf
import numpy as np
from label_db import LabelDB

def img_xys_iterator(base_dir, batch_size, patch_fraction):
  dataset = tf.data.Dataset.from_generator(partial(_decode_img_and_lookup_xys,
                                                   base_dir,
                                                   for_training=False,
                                                   patch_fraction=patch_fraction),
                                           output_types=(tf.uint8, tf.uint8))
  dataset = dataset.batch(batch_size) #.prefetch(2)
  return dataset.make_one_shot_iterator().get_next()

def _decode_img_and_lookup_xys(base_dir, for_training, patch_fraction):
  label_db = LabelDB(check_same_thread=False)
  while True:
    for fname in os.listdir(base_dir):

      # load full size image
      img = Image.open("%s/%s" % (base_dir, fname))
      w, h = img.size
      print "h", h, "w", w
      # TODO: drop use of img_shape completely
      
      # TODO: distort here
  
      # TODO: lookup xys
      xys = label_db.get_labels(fname)
      print("xys", fname, xys)

      # sample patch
      if patch_fraction >= 1:
        # decide patch bounds
        pw, ph = w / patch_fraction, h / patch_fraction
        x1 = 0 #int(random.random() * (w - pw))
        y1 = 0 #int(random.random() * (h - ph))
        x2 = x1 + pw
        y2 = y1 + ph
        print("PATCH!", x1, y1, x2, y2)
        # crop image
        img = img.crop((x1, y1, x2, y2))
        print "post crop", img.size
        # recreate xys
        patch_xys = []
        for x, y in xys:
          # note: x, y as saved in db are for numpy arrays, not PIL images
          # to deal with pil images we need to swap them
          print("check", x, y)
#          x = int(x / patch_fraction)
#          y = int(y / patch_fraction)
#          print("remaps to", x, y)
          if x > x1 and x < x2 and y > y1 and y < y2:
            print("..in patch as", x-x1, y-y1)
            patch_xys.append((x-x1, y-y1))
        xys = patch_xys
        h, w = ph, pw
        print "xyx post patching", xys
        
#      if img_shape == '256':
#        xys = [(33, 26), (162, 143), (169, 221)]
#      elif img_shape == '48':
#        xys = [(29, 46), (29, 37), (15, 23)]
#      elif img_shape == '384':
#        xys = [(242, 302), (116, 182), (251, 377), (92, 79), (69, 149), (60, 109), (56, 150), (64, 131), (84, 145)]
#      else:
#        raise Exception("??? %s" % img_shape)
  
      # done
      yield np.array(img), xys_to_bitmap(xys, h, w, rescale=0.5)

def xys_to_bitmap(xys, height, width, rescale=1.0):
  print ">xys_to_bitmap xys", xys, "height", height, "width", width, "rescale", rescale
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  print "bitmap.shape", bitmap.shape
  for x, y in xys:
    print "set", int(x*rescale), int(y*rescale)
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
  parser.add_argument('--image-dir', type=str, default="images/sample_originals")
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--patch-fraction', type=int, default=1,
                      help="what fraction of image to use as patch. 1 => no patch")
  opts = parser.parse_args()
  
  from PIL import Image, ImageDraw

  sess = tf.Session()
  
  imgs, xys = img_xys_iterator(base_dir=opts.image_dir,
                               batch_size=opts.batch_size,
                               patch_fraction=opts.patch_fraction)
  
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
    
    


