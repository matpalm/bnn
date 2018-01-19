#!/usr/bin/env python

import numpy as np
import tensorflow as tf

np.set_printoptions(precision=2, threshold=10000, suppress=True, linewidth=10000)

def dice_loss_2class(y, y_hat, smoothing=0):
  assert y.shape == y_hat.shape
  # calculate intersection by using the true values to act as a mask against the y_hat values.
  intersection = y * y_hat
  # for binary classification we can do sum across axis=1
  score = 2.0 * (intersection.sum(1) + smoothing) / (y.sum(1) + y_hat.sum(1) + smoothing)
  loss = 1.0 - np.mean(score)
  return loss

def dice_loss(y, y_hat, smoothing=0):
  y = np.array(y)
  y_hat = np.array(y_hat)
  assert y.shape == y_hat.shape
  # calculate intersection by using the true values to act as a mask against the y_hat values.
  intersection = y * y_hat
  print "intersection", intersection, intersection.sum()
  print "nom", intersection.sum() + smoothing
  print "denom", (y.sum() + y_hat.sum() + smoothing)
  # for binary classification we can do sum across axis=1
  score = 2.0 * (intersection.sum() + smoothing) / (y.sum() + y_hat.sum() + smoothing)
  loss = 1.0 - score
  return loss

print "NP dice_loss eg1"
print dice_loss(y     = [1,1,0,1,0],
                y_hat = [1,1,0,1,0],
                smoothing=0.001)
print "NP dice_loss eg2"
print dice_loss(y     = [1,1,0,1,0],
                y_hat = [0.9,0.9,0.01,0.9,0.1],
                smoothing=0.001)
print "NP dice_loss eg3"
print dice_loss(y     = [1,1,0,1,0],
                y_hat = [0,0,1,0,1],
                smoothing=0.001)


def dice_loss_tf(y, y_hat, smoothing=0):
  intersection = y * y_hat
  intersection_rs = tf.reduce_sum(intersection, axis=1)
  nom = intersection_rs + smoothing
  denom = tf.reduce_sum(y, axis=1) + tf.reduce_sum(y_hat, axis=1) + smoothing
  score = 2.0 * (nom / denom)
  loss = 1.0 - score
#  loss = tf.Print(loss, [intersection, intersection_rs, nom, denom], first_n=100, summarize=10000)
  return loss

y = tf.placeholder(tf.float32)
y_hat = tf.placeholder(tf.float32)
dl = dice_loss_tf(y, y_hat, smoothing=1e-4)

with tf.Session() as s:
  print "TF dice loss"
  print s.run(dl, feed_dict={y: [[1, 1, 0, 1, 0, 0],
                                 [1, 1, 0, 1, 0, 0],
                                 [1, 1, 0, 1, 0, 0]],
                             y_hat: [[1, 1, 0, 1, 0, 0],
                                     [0.9, 0.9, 0.01, 0.9, 0.1, 0.1],
                                     [0, 0, 1, 0, 1, 1]]})

