#!/usr/bin/env python3

import model
import tensorflow as tf
import numpy as np
#import argparse
from tensorflow.python.tools import inspect_checkpoint as ic
import util as u
from PIL import Image

# setup random initial images; 64 single colours
initial_imgs = np.empty((64, 16, 16, 3), dtype=np.float32)
for idx in range(64):
  colour = np.random.uniform(-1, 1, size=(3))
  initial_imgs[idx] = np.tile(colour, [16,16,1])
imgs = tf.get_variable(name="imgs", dtype=tf.float32,
                       initializer=tf.constant(initial_imgs))

# target is black image with single pixel lit (i.e. one bee in middle)
target_bitmap = np.zeros((8, 8, 1)).astype(np.float32)
target_bitmap[3, 3] = 1.0
# tiled along batch dim
target_bitmap = np.tile(np.expand_dims(target_bitmap, 0), [64, 1, 1, 1])

# init e3b model
model = model.Model(imgs,
                    is_training=False,
                    use_skip_connections=True,
                    base_filter_size=4,
                    use_batch_norm=False)

# restore model
# we can't use the model.restore since there is now one extra variable that
# before was a placeholder; imgs. if we try to restore with a Saver() it'll
# complain the ckpt doesn't have that variable. so instead create a saver
# that includes _just_ the variables in the model, i.e. doesn't include
# the 'img' variable, and then init that one explicitly. #clumsy
model_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope="train_test_model")
saver = tf.train.Saver(var_list=model_variables)
sess = tf.Session()
ckpt_file = tf.train.latest_checkpoint("ckpts/e3b")
saver.restore(sess, ckpt_file)
sess.run(imgs.initializer)

# define optimisation using xent against _just_ the input images
loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=target_bitmap,
                                            logits=model.logits))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
# TODO: try this, will need to init opt specific vars
#optimiser = tf.train.AdamOptimizer(learning_rate=1e-2)
train_op = optimiser.minimize(loss, var_list=[imgs])

def dump_images(prefix):
  # run from imgs -> bitmap and stitch them together...
  img_collage = Image.new('RGB', (17*8, 17*8), (0, 0, 0))
  bitmap_collage = Image.new('RGB', (9*8, 9*8), (255, 255, 255))
  centroids_collage = Image.new('RGB', (9*8, 9*8), (255, 255, 255))
  ims, bs = sess.run([imgs, model.output])
  for x in range(8):
    for y in range(8):
      i = (x * 8) + y
      img_collage.paste(u.zero_centered_array_to_pil_image(ims[i]), (17*x, 17*y))
      output_bitmap = u.bitmap_to_pil_image(bs[i])
      bitmap_collage.paste(output_bitmap, (9*x, 9*y))
      centroids = u.centroids_of_connected_components(bs[i])
      centroid_bitmap = u.bitmap_from_centroids(centroids, h=8, w=8)
      centroid_bitmap = u.bitmap_to_single_channel_pil_image(centroid_bitmap)
      centroids_collage.paste(centroid_bitmap, (9*x, 9*y))
  img_collage.save("images/ra/%s_imgs.png" % prefix)
  bitmap_collage.save("images/ra/%s_bitmaps.png" % prefix)
  centroids_collage.save("images/ra/%s_centroids.png" % prefix)

dump_images("start")

n = 0
while True:
  # run optimisation for a bit...
  for i in range(1000):
    l, _ = sess.run([loss, train_op])
    print(i, l)
  dump_images("%03d" % n)
  n += 1
