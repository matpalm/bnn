from __future__ import print_function

import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

def dump_shape_and_product_of(tag, t):
  shape_product = 1
  for dim in t.get_shape().as_list()[1:]:
    shape_product *= dim
  print("%-10s %-20s #%s" % (tag, t.get_shape(), shape_product), file=sys.stderr)

class Model(object):

  def __init__(self, imgs):
    self.is_training = tf.placeholder(tf.bool, name="is_training")
    self.imgs = imgs

    # leave this conversion as late as possible to avoid storing floats in queue
    # TODO: do we actually care?
    model = tf.image.convert_image_dtype(self.imgs, dtype=tf.float32)

    # center from [0, 1] to [-1, 1]
    # TODO: this is habit; do we need it?
    model = (model * 2) - 1
    dump_shape_and_product_of('input', model)

    # TODO: add skip connections

    model = slim.conv2d(model, num_outputs=8, kernel_size=3, stride=2, scope='e1')
    dump_shape_and_product_of('e1', model)
    
    model = slim.conv2d(model, num_outputs=16, kernel_size=3, stride=2, scope='e2')
    dump_shape_and_product_of('e2', model)
    
    model = slim.conv2d(model, num_outputs=32, kernel_size=3, stride=2, scope='e3')
    dump_shape_and_product_of('e3', model)
    
    model = slim.conv2d(model, num_outputs=32, kernel_size=3, stride=2, scope='e4')
    dump_shape_and_product_of('e3', model)

    # record bottlenecked shape for resizing back
    # this is clumsy, how to do this more directly from tensors / config?
    shape = model.get_shape().as_list()[1:]
    h, w = shape[0], shape[1]
    
    model = tf.image.resize_nearest_neighbor(model, [h*2, w*2])
    model = slim.conv2d(model, num_outputs=32, kernel_size=4, scope='d1')
    dump_shape_and_product_of('d1', model)

    model = tf.image.resize_nearest_neighbor(model, [h*4, w*4])
    model = slim.conv2d(model, num_outputs=16, kernel_size=4, scope='d2')
    dump_shape_and_product_of('d2', model)

    model = tf.image.resize_nearest_neighbor(model, [h*8, w*8])
    model = slim.conv2d(model, num_outputs=8, kernel_size=4, scope='d3')
    dump_shape_and_product_of('d3', model)

    # at this point we are back to 1/2 res of original, which should be enough
    
    # finally mapping to binary
    self.logits = slim.conv2d(model, num_outputs=1, kernel_size=3, scope='d4',
                              activation_fn=None)
    dump_shape_and_product_of('logits', self.logits)  # (16, 16, 1)

    self.output = tf.nn.sigmoid(self.logits)
    # TODO what should be applied here as activation?    

    
    
