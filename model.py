from __future__ import print_function

import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import os

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)

class Model(object):

  def __init__(self, imgs, use_skip_connections=True, base_filter_size=16):
    print("use_skip_connections", use_skip_connections)
    self.is_training = tf.placeholder(tf.bool, name="is_training")
    self.imgs = imgs

    # leave this conversion as late as possible to avoid storing floats in queue
    # TODO: do we actually care?
    model = tf.image.convert_image_dtype(self.imgs, dtype=tf.float32)

    # center from [0, 1] to [-1, 1]
    # TODO: this is habit; do we need it?
    model = (model * 2) - 1
    dump_shape_and_product_of('input', model)
    
    e1 = slim.conv2d(model, num_outputs=base_filter_size, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', e1)
    
    e2 = slim.conv2d(e1, num_outputs=2*base_filter_size, kernel_size=3, stride=2, scope='e2')
    dump_shape_and_product_of('e2', e2)
    
    e3 = slim.conv2d(e2, num_outputs=4*base_filter_size, kernel_size=3, stride=2, scope='e3')
    dump_shape_and_product_of('e3', e3)
    
    e4 = slim.conv2d(e3, num_outputs=8*base_filter_size, kernel_size=3, stride=2, scope='e4')
    dump_shape_and_product_of('e4', e4)

    # record bottlenecked shape for resizing back
    # this is clumsy, how to do this more directly from tensors / config?
    shape = e4.get_shape().as_list()[1:]
    h, w = shape[0], shape[1]
    
    model = tf.image.resize_nearest_neighbor(e4, [h*2, w*2])
    model = slim.conv2d(model, num_outputs=4*base_filter_size, kernel_size=3, scope='d1')
    dump_shape_and_product_of('d1', model)

    if use_skip_connections:
      model = tf.concat([model, e3], axis=3)
      dump_shape_and_product_of('d1+e3', model)

    model = tf.image.resize_nearest_neighbor(model, [h*4, w*4])
    model = slim.conv2d(model, num_outputs=2*base_filter_size, kernel_size=3, scope='d2')
    dump_shape_and_product_of('d2', model)

    if use_skip_connections:
      model = tf.concat([model, e2], axis=3)
      dump_shape_and_product_of('d2+e2', model)

    model = tf.image.resize_nearest_neighbor(model, [h*8, w*8])
    model = slim.conv2d(model, num_outputs=base_filter_size, kernel_size=3, scope='d3')
    dump_shape_and_product_of('d3', model)

    if use_skip_connections:
      model = tf.concat([model, e1], axis=3)
      dump_shape_and_product_of('d3+e1', model)

    # at this point we are back to 1/2 res of original, which should be enough
    
    # finally mapping to binary
    self.logits = slim.conv2d(model, num_outputs=1, kernel_size=3, scope='d4',
                              activation_fn=None)
    dump_shape_and_product_of('logits', self.logits)

    self.output = tf.nn.sigmoid(self.logits)

    self.saver = tf.train.Saver(max_to_keep=100,
                                keep_checkpoint_every_n_hours=1)

  def save(self, sess, ckpt_dir):
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    dts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_ckpt = "%s/%s" % (ckpt_dir, dts)
    self.saver.save(sess, new_ckpt)

  def restore(self, sess, ckpt_dir, ckpt_file=None):
    if ckpt_file is None:
      ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    else:
      ckpt_file = "%s/%s" % (ckpt_dir, ckpt_file)
    self.saver.restore(sess, ckpt_file)
