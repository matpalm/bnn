from PIL import Image, ImageDraw
from skimage import measure
import io
import math
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

def hms(secs):
  if secs < 0:
    return "<0"  # clumsy
  secs = int(secs)
  mins, secs = divmod(secs, 60)
  hrs, mins = divmod(mins, 60)
  if hrs > 0:
    return "%d:%02d:%02d" % (hrs, mins, secs)
  elif mins > 0:
    return "%02d:%02d" % (mins, secs)
  else:
    return "%02d" % secs

def xys_to_bitmap(xys, height, width, rescale=1.0):
  # note: include trailing 1 dim to easier match model output
  bitmap = np.zeros((int(height*rescale), int(width*rescale), 1), dtype=np.float32)
  for x, y in xys:
    try:
      bitmap[int(y*rescale), int(x*rescale), 0] = 1.0  # recall images are (height, width)
    except IndexError as e:
      print("IndexError: are --height and --width correct?")
      raise e
  return bitmap

def debug_img(img, bitmap, logistic_output):
  # create a debug image with three columns; 1) original RGB. 2) black/white
  # bitmap of labels 3) black/white bitmap of predictions (with centroids coloured
  # red.
  h, w, _channels = bitmap.shape
  canvas = Image.new('RGB', (w*3, h), (50, 50, 50))
  # original input image on left
  img = zero_centered_array_to_pil_image(img)
  img = img.resize((w, h))
  canvas.paste(img, (0, 0))
  # label bitmap in center
  canvas.paste(bitmap_to_pil_image(bitmap), (w, 0))
  # logistic output on right
  canvas.paste(bitmap_to_pil_image(logistic_output), (w*2, 0))
  # draw red dots on right hand side image corresponding to
  # final thresholded prediction
  draw = ImageDraw.Draw(canvas)
  for y, x in centroids_of_connected_components(logistic_output):
    draw.rectangle((w*2+x,y,w*2+x,y), fill='red')
  # finally draw blue lines between the three to delimit boundaries
  draw.line([w,0,w,h], fill='blue')
  draw.line([2*w,0,2*w,h], fill='blue')
  draw.line([3*w,0,3*w,h], fill='blue')
  # done
  return canvas

def explicit_summaries(tag_values):
  values = [tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_values.items()]
  return tf.Summary(value=values)

def pil_image_to_tf_summary(img, tag="debug_img"):
  # serialise png bytes
  sio = io.BytesIO()
  img.save(sio, format="png")
  png_bytes = sio.getvalue()

  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
  return tf.Summary(value=[tf.Summary.Value(tag=tag,
                                            image=tf.Summary.Image(height=img.size[0],
                                                                   width=img.size[1],
                                                                   colorspace=3, # RGB
                                                                   encoded_image_string=png_bytes))])

#def dice_loss(y, y_hat, batch_size, smoothing=0):
#  y = tf.reshape(y, (batch_size, -1))
#  y_hat = tf.reshape(y_hat, (batch_size, -1))
#  intersection = y * y_hat
#  intersection_rs = tf.reduce_sum(intersection, axis=1)
#  nom = intersection_rs + smoothing
#  denom = tf.reduce_sum(y, axis=1) + tf.reduce_sum(y_hat, axis=1) + smoothing
#  score = 2.0 * (nom / denom)
#  loss = 1.0 - score
#  loss = tf.Print(loss, [intersection, intersection_rs, nom, denom], first_n=100, summarize=10000)
#  return loss

def centroids_of_connected_components(bitmap, threshold=0.05, rescale=1.0):
  # TODO: don't do raw (binary) threshold; instead use P(y) as weighting for centroid
  #       e.g. https://arxiv.org/abs/1806.03413 sec 3.D
  #       update: this didn't help much :/ centroid weighted by intensities moved only up
  #       to a single pixel (guess centroids are already quite evenly dispersed)
  #       see https://gist.github.com/matpalm/20a3974ceb7f632f935285262fac4e98
  # TODO: hunt down the x/y swap between PIL and label db :/

  # threshold
  mask = bitmap > threshold
  bitmap = np.zeros_like(bitmap)
  bitmap[mask] = 1.0
  # calc connected components
  all_labels = measure.label(bitmap)
  # return centroids
  centroids = []
  for region in measure.regionprops(label_image=all_labels):
    cx, cy = map(lambda p: int(p*rescale), (region.centroid[0], region.centroid[1]))
    centroids.append((cx, cy))
  return centroids

def bitmap_from_centroids(centroids, h, w):
  bitmap = np.zeros((h, w, 1))
  for cx, cy in centroids:
    bitmap[cx, cy] = 1.0
  return bitmap

def zero_centered_array_to_pil_image(orig_array):
  assert orig_array.dtype == np.float32
  h, w, c = orig_array.shape
  assert c == 3
  array = orig_array + 1  # 0.0 -> 2.0
  array *= 127.5  # 0.0 -> 255.0
  array = array.copy().astype(np.uint8)
  assert np.min(array) >= 0
  assert np.max(array) <= 255
  return Image.fromarray(array)

def bitmap_to_pil_image(bitmap):
  assert bitmap.dtype == np.float32
  h, w, c = bitmap.shape
  assert c == 1
  rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
  single_channel = bitmap[:,:,0] * 255
  rgb_array[:,:,0] = single_channel
  rgb_array[:,:,1] = single_channel
  rgb_array[:,:,2] = single_channel
  return Image.fromarray(rgb_array)

def bitmap_to_single_channel_pil_image(bitmap):
  h, w, c = bitmap.shape
  assert c == 1
  bitmap = np.uint8(bitmap[:,:,0] * 255)
  return Image.fromarray(bitmap, mode='L')  # L => (8-bit pixels, black and white)

def side_by_side(rgb, bitmap):
  h, w, _ = rgb.shape
  canvas = Image.new('RGB', (w*2, h), (50, 50, 50))
  # paste RGB on left hand side
  lhs = zero_centered_array_to_pil_image(rgb)
  canvas.paste(lhs, (0, 0))
  # paste bitmap version of labels on right hand side
  # black with white dots at labels
  rhs = bitmap_to_pil_image(bitmap)
  rhs = rhs.resize((w, h))
  canvas.paste(rhs, (w, 0))
  # draw on a blue border (and blue middle divider) to make it
  # easier to see relative positions.
  draw = ImageDraw.Draw(canvas)
  draw.polygon([0,0,w*2-1,0,w*2-1,h-1,0,h-1], outline='blue')
  draw.line([w,0,w,h], fill='blue')
  canvas = canvas.resize((w, h//2))
  return canvas

def red_dots(rgb, centroids):
  img = zero_centered_array_to_pil_image(rgb)
  canvas = ImageDraw.Draw(img)
  for y, x in centroids:  # recall: x/y flipped between db & pil
    canvas.rectangle((x-2,y-2,x+2,y+2), fill='red')
  return img


class SetComparison(object):
  def __init__(self):
    self.true_positive_count = 0
    self.false_negative_count = 0
    self.false_positive_count = 0

  def compare_sets(self, true_pts, predicted_pts, threshold=10.0):
    # compare two sets of true & predicted centroids and calculate TP, FP and FN rate.

    # iteratively find closest point in each set and if they are close enough (according
    # to threshold) declare them them a match (i.e. true positive). once the closest
    # match is above the threshold, or we run out of points to match, stop comparing.
    # whatever remains in true_pts & predicted_pts after matching is done are false
    # negatives & positives respectively.
    TP = 0
    while len(true_pts) > 0 and len(predicted_pts) > 0:
      # find indexes of closest pair
      closest_pair = None
      closest_sqr_distance = None
      for t_i, t in enumerate(true_pts):
        for p_i, p in enumerate(predicted_pts):
          sqr_distance = (t[0]-p[0])**2 + (t[1]-p[1])**2
          if closest_sqr_distance is None or sqr_distance < closest_sqr_distance:
            closest_pair = t_i, p_i
            closest_sqr_distance = sqr_distance
      # if closest pair is above threshold so comparing
      closest_distance = math.sqrt(closest_sqr_distance)
      if closest_distance > threshold:
        break
      # otherwise delete closest pair & declare them a match
      t_i, p_i = closest_pair
      del true_pts[t_i]
      del predicted_pts[p_i]
      TP += 1

    # remaining unmatched entries are false positives & negatives.
    FN = len(true_pts)
    FP = len(predicted_pts)

    # aggregate
    self.true_positive_count += TP
    self.false_negative_count += FN
    self.false_positive_count += FP

    # return for just this comparison
    return TP, FN, FP

  def precision_recall_f1(self):
    try:
      precision = self.true_positive_count / (self.true_positive_count + self.false_positive_count)
      recall = self.true_positive_count / (self.true_positive_count + self.false_negative_count)
      f1 = 2 * (precision * recall) / (precision + recall)
      return precision, recall, f1
    except ZeroDivisionError:
      return 0, 0, 0

def check_images(fnames):
  prev_width, prev_height = 0, 0
  for i, fname in enumerate(fnames):
    try:
      im = Image.open(fname)
    except IOError as e:
      print("Image is corrupted or does not exist:", fname)
      raise e
    width, height = im.size
    if i == 0:
      prev_width = width
      prev_height = height
    elif not prev_width == width or not prev_height == height:
      print("Image size does not match others:", fname, "wh:", width, height)
      exit()
  return width, height

def latest_checkpoint_in_dir(ckpt_dir):
  checkpoint_info = yaml.load(open("%s/checkpoint" % ckpt_dir).read())
  return checkpoint_info['model_checkpoint_path']
