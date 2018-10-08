from __future__ import print_function

import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import os
import util as u

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)

class Model(object):

  def __init__(self, imgs, is_training, use_skip_connections,
               base_filter_size, use_batch_norm):
    with tf.variable_scope("train_test_model", reuse=tf.AUTO_REUSE) as scope:
      self.imgs = imgs
      model = imgs

      dump_shape_and_product_of('input', model)

      e1 = slim.conv2d(model, num_outputs=base_filter_size, kernel_size=3, stride=2, scope='e1')
      if use_batch_norm:
        e1 = slim.batch_norm(e1, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('e1', e1)

      e2 = slim.conv2d(e1, num_outputs=2*base_filter_size, kernel_size=3, stride=2, scope='e2')
      if use_batch_norm:
        e2 = slim.batch_norm(e2, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('e2', e2)

      e3 = slim.conv2d(e2, num_outputs=4*base_filter_size, kernel_size=3, stride=2, scope='e3')
      if use_batch_norm:
        e3 = slim.batch_norm(e3, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('e3', e3)

      e4 = slim.conv2d(e3, num_outputs=8*base_filter_size, kernel_size=3, stride=2, scope='e4')
      if use_batch_norm:
        e4 = slim.batch_norm(e4, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('e4', e4)

      # record bottlenecked shape for resizing back
      # this is clumsy, how to do this more directly from tensors / config?
      _batch_size, h, w, _depth = e4.get_shape().as_list()

      model = tf.image.resize_nearest_neighbor(e4, [h*2, w*2])
      model = slim.conv2d(model, num_outputs=4*base_filter_size, kernel_size=3, scope='d1')
      if use_batch_norm:
        model = slim.batch_norm(model, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('d1', model)

      if use_skip_connections:
        model = tf.concat([model, e3], axis=3)
        dump_shape_and_product_of('d1+e3', model)

      model = tf.image.resize_nearest_neighbor(model, [h*4, w*4])
      model = slim.conv2d(model, num_outputs=2*base_filter_size, kernel_size=3, scope='d2')
      if use_batch_norm:
        model = slim.batch_norm(model, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('d2', model)

      if use_skip_connections:
        model = tf.concat([model, e2], axis=3)
        dump_shape_and_product_of('d2+e2', model)

      model = tf.image.resize_nearest_neighbor(model, [h*8, w*8])
      model = slim.conv2d(model, num_outputs=base_filter_size, kernel_size=3, scope='d3')
      if use_batch_norm:
        model = slim.batch_norm(model, decay=0.9, is_training=is_training)
      dump_shape_and_product_of('d3', model)

      if use_skip_connections:
        model = tf.concat([model, e1], axis=3)
        dump_shape_and_product_of('d3+e1', model)

      # at this point we are back to 1/2 res of original, which should be enough

      # finally mapping to binary
      self.logits = slim.conv2d(model, num_outputs=1, kernel_size=3, scope='d4',
                                activation_fn=None)
      dump_shape_and_product_of('logits', self.logits)

      self.output = tf.nn.sigmoid(self.logits, name='output')

      self.saver = tf.train.Saver(max_to_keep=100,
                                  keep_checkpoint_every_n_hours=1)

  def calculate_losses_wrt(self, labels, pos_weight=1.0):
    self.xent_loss = tf.reduce_mean(
      tf.nn.weighted_cross_entropy_with_logits(targets=labels,
                                               logits=self.logits,
                                               pos_weight=pos_weight))

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



if __name__ == "__main__":
  # build model just to get debug on sizes
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--width', type=int, default=768, help='input image width')
  parser.add_argument('--height', type=int, default=1024, help='input image height')
  parser.add_argument('--no-use-skip-connections', action='store_true')
  parser.add_argument('--no-use-batch-norm', action='store_true')
  parser.add_argument('--base-filter-size', type=int, default=8)
  opts = parser.parse_args()

  Model(imgs=tf.placeholder(dtype=tf.float32, shape=(1, opts.width, opts.height, 3), name='input_imgs'),
        is_training=False,
        use_skip_connections=not opts.no_use_skip_connections,
        base_filter_size=opts.base_filter_size,
        use_batch_norm=not opts.no_use_batch_norm)
