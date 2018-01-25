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
parser.add_argument('--train-image-dir', type=str, default="images/sample_originals/train")
parser.add_argument('--test-image-dir', type=str, default="images/sample_originals/test")
#parser.add_argument('--train-data', type=str, default="train.jsons",
#                    help="json file for training data")
# parser.add_argument('--test-data', type=str, default="test.jsons",
#                     help="json file for training data")
#parser.add_argument('--run', type=str, required=True)
#parser.add_argument('--img-shape', type=str, default='384')
#parser.add_argument('--steps', type=int, default=10,
#                    help='number of x100 training + test steps')
parser.add_argument('--patch-fraction', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=0.01)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)
  
np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

# Build readers for train and test data.
train_imgs, train_xys_bitmaps = data.img_xys_iterator(base_dir=opts.train_image_dir,
                                                      batch_size=opts.batch_size,
                                                      patch_fraction=opts.patch_fraction)
test_imgs, test_xys_bitmaps = data.img_xys_iterator(base_dir=opts.test_image_dir,
                                                    batch_size=1,
                                                    patch_fraction=1)
# TODO! divide by patch_fraction here instead of calculating explicitly.
train_imgs = tf.reshape(train_imgs, (opts.batch_size, 1024/opts.patch_fraction, 768/opts.patch_fraction, 3))  
test_imgs = tf.reshape(test_imgs, (1, 1024, 768, 3))

# Build training and test model with same params.
with tf.variable_scope("train_test_model") as scope:
  print("patch train model...")
  train_model = model.Model(train_imgs)
  scope.reuse_variables()
  print("full res test model...")  
  test_model = model.Model(test_imgs)

# Setup loss and optimiser.
labels = tf.to_float(train_xys_bitmaps)

xent_loss = tf.reduce_mean(
  tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                          logits=train_model.logits))

def calc_dice_loss(y, y_hat, smoothing=0):
  y = tf.reshape(y, (opts.batch_size, -1))
  y_hat = tf.reshape(y_hat, (opts.batch_size, -1))
  intersection = y * y_hat
  intersection_rs = tf.reduce_sum(intersection, axis=1)
  nom = intersection_rs + smoothing
  denom = tf.reduce_sum(y, axis=1) + tf.reduce_sum(y_hat, axis=1) + smoothing
  score = 2.0 * (nom / denom)
  loss = 1.0 - score
#  loss = tf.Print(loss, [intersection, intersection_rs, nom, denom], first_n=100, summarize=10000)
  return loss

dice_loss = tf.reduce_mean(calc_dice_loss(labels,
                                          tf.sigmoid(train_model.logits),
                                          smoothing=1e-2))

optimiser = tf.train.AdamOptimizer(learning_rate=opts.learning_rate)
xent_weight = 1.000
dice_weight = 0.000
train_op = optimiser.minimize(xent_weight * xent_loss + dice_weight * dice_loss)


#train_op = slim.learning.create_train_op(total_loss = loss, # + regularisation_loss,
#                                         optimizer = optimiser,
#                                         summarize_gradients = False)

# Create session.
sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=sess_config)
global_step = tf.train.create_global_step()
sess.run(tf.global_variables_initializer())

# Setup summaries.
train_summaries = [
  tf.summary.scalar('xent_loss', xent_loss),
  tf.summary.scalar('dice_loss', dice_loss),
]
train_summaries_op = tf.summary.merge(train_summaries)
train_summaries_writer = tf.summary.FileWriter("tb/training", sess.graph)

for idx in range(10000):
  print("---------------------------------", idx)

  # train a bit.
  for _ in range(20):
    _, xl, dl = sess.run([train_op, xent_loss, dice_loss],
                         feed_dict={train_model.is_training: True})
    print(xl, dl)

  # one additional step of training for summaries
  _, summaries, step = sess.run([train_op, train_summaries_op, global_step],
                                feed_dict={train_model.is_training: True})
  train_summaries_writer.add_summary(summaries, idx)

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
  debug_img(i, bm, o).save("train_%05d.png" % idx)
  
  # dump an image from held out test image (full res)
  i, bm, o = sess.run([test_imgs, test_xys_bitmaps, test_model.output],
                      feed_dict={test_model.is_training: False})
  debug_img(i, bm, o).save("test_%05d.png" % idx)






