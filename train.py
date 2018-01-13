#!/usr/bin/env python
from __future__ import print_function

import argparse
from sklearn.metrics import confusion_matrix
import data
import model
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, default="images/sample_originals")              
#parser.add_argument('--train-data', type=str, default="train.jsons",
#                    help="json file for training data")
# parser.add_argument('--test-data', type=str, default="test.jsons",
#                     help="json file for training data")
#parser.add_argument('--run', type=str, required=True)
#parser.add_argument('--img-shape', type=str, default='384')
#parser.add_argument('--steps', type=int, default=10,
#                    help='number of x100 training + test steps')
#parser.add_argument('--num-filters', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.01)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)
  
np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

train_imgs, train_xys_bitmaps = data.img_xys_iterator(base_dir=opts.image_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_fraction=2)
test_imgs, test_xys_bitmaps = data.img_xys_iterator(base_dir=opts.image_dir,   # TODO: different test set
                                                    batch_size=1,
                                                    patch_fraction=1)

# TODO! divide by patch_fraction here instead of calculating explicitly.
train_imgs = tf.reshape(train_imgs, (opts.batch_size, 512, 384, 3))  
test_imgs = tf.reshape(test_imgs, (1, 1024, 768, 3))

with tf.variable_scope("train_test_model") as scope:
  print("patch train model...")
  train_model = model.Model(train_imgs)
  scope.reuse_variables()
  print("full res test model...")  
  test_model = model.Model(test_imgs)

global_step = tf.train.create_global_step()

print("xys_bitmaps", train_xys_bitmaps.shape)
print("logits", train_model.logits.shape)

loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(train_xys_bitmaps),
                                            logits=train_model.logits))
#loss = tf.reduce_mean(tf.nn.l2_loss(xys_bitmaps - model.output))

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
train_op = optimiser.minimize(loss)

#train_op = slim.learning.create_train_op(total_loss = loss, # + regularisation_loss,
#                                         optimizer = optimiser,
#                                         summarize_gradients = False)

sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=sess_config)

sess.run(tf.global_variables_initializer())

for idx in range(200):
  print("---------------------------------", idx)

  # train a bit.
  for _ in range(100):
    _, l = sess.run([train_op, loss],
                    feed_dict={train_model.is_training: True})
  print(l)

  # dump images from train and test.
  # TODO: move to tensorboard

  from PIL import Image, ImageDraw
  def debug_img(i, bm, o):
    _bs, h, w, _c = bm.shape
    canvas = Image.new('RGB', (w*3, h), (50, 50, 50))
    i = Image.fromarray(i[0])
    i = i.resize((w, h))  
    canvas.paste(i, (0, 0))
    bm = data.bitmap_to_pil_image(bm[0])
    canvas.paste(bm, (w, 0))
    o = data.bitmap_to_pil_image(o[0])
    canvas.paste(o, (w*2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([w,0,w,h], fill='blue')
    draw.line([2*w,0,2*w,h], fill='blue')
    draw.line([3*w,0,3*w,h], fill='blue')
    return canvas
    
  # dump an image of training with a) RGB b) labels c) predictions
  # recall: RGB twice size of labelled and predicted.  
  i, bm, o = sess.run([train_imgs, train_xys_bitmaps, train_model.output],
                      feed_dict={train_model.is_training: False})
  print("train")
  print("i", i.shape, np.min(i), np.mean(i), np.max(i))
  print("bm", bm.shape, np.min(bm), np.mean(bm), np.max(bm))
#  print("l", l.shape, np.min(l), np.mean(l), np.max(l))
  print("o", o.shape, np.min(o), np.mean(o), np.max(o))
  debug_img(i, bm, o).save("train_%03d.png" % idx)
  
  # dump an image from held out test image (full res)
  # TODO: move to tensorboard
  i, bm, o = sess.run([test_imgs, test_xys_bitmaps, test_model.output],
                      feed_dict={test_model.is_training: False})
  print("test")
  print("i", i.shape, np.min(i), np.mean(i), np.max(i))
  print("bm", bm.shape, np.min(bm), np.mean(bm), np.max(bm))
#  print("l", l.shape, np.min(l), np.mean(l), np.max(l))
  print("o", o.shape, np.min(o), np.mean(o), np.max(o))
  debug_img(i, bm, o).save("test_%03d.png" % idx)






