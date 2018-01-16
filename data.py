#!/usr/bin/env python
from PIL import Image, ImageDraw
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
  dataset = dataset.batch(batch_size).prefetch(4)
  return dataset.make_one_shot_iterator().get_next()

def _decode_img_and_lookup_xys(base_dir, for_training, patch_fraction):
  label_db = LabelDB(check_same_thread=False)
  while True:
    for fname in os.listdir(base_dir):

      # load full size image
      img = Image.open("%s/%s" % (base_dir, fname))
      w, h = img.size
      
      # TODO: distort here
  
      # lookup xys
      xys = label_db.get_labels(fname)

      # sample a random patch (if required)
      if patch_fraction > 1:
        # decide patch bounds
        pw, ph = w / patch_fraction, h / patch_fraction
        x1 = int(random.random() * (w - pw))
        y1 = int(random.random() * (h - ph))
        x2 = x1 + pw
        y2 = y1 + ph
        # crop image
        img = img.crop((x1, y1, x2, y2))
        # recreate xys
        patch_xys = []
        for x, y in xys:
          # note: x, y as saved in db are for numpy arrays, not PIL images
          # to deal with pil images we need to swap them
          if x > x1 and x < x2 and y > y1 and y < y2:
            patch_xys.append((x-x1, y-y1))
        xys = patch_xys
        h, w = ph, pw
        
      # done
      yield np.array(img), xys_to_bitmap(xys, h, w, rescale=0.5)

def xys_to_bitmap(xys, height, width, rescale=1.0):
  # note: include trailing 1 dim to easier match model output
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  for x, y in xys:
    bitmap[int(y*rescale), int(x*rescale), 0] = 1.0  # recall images are (height, width)
  return bitmap

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
  opts = parser.parse_args()
  
  from PIL import Image, ImageDraw

  sess = tf.Session()
  
  imgs, xys = img_xys_iterator(base_dir=opts.image_dir,
                               batch_size=opts.batch_size,
                               patch_fraction=opts.patch_fraction)
  
  img_batch, xys_batch = sess.run([imgs, xys])
  for i, (img, xys) in enumerate(zip(img_batch, xys_batch)):
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

    # Draw on a blue border (and blue middle divider) to make it
    # easier to see relative positions.
    draw = ImageDraw.Draw(canvas)
    draw.polygon([0,0,w*2-1,0,w*2-1,h-1,0,h-1], outline='blue')
    draw.line([w,0,w,h], fill='blue')
    
    canvas.save("test_%d.png" % i)
    
    


