from PIL import Image, ImageDraw
import StringIO
import data
import tensorflow as tf

def debug_img(i, bm, o):
  # create a debug image with three columns; 1) original RGB. 2) black/white
  # bitmap of labels 3) black/white bitmap of predictions
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

def PILImageToTFSummary(img):
  # serialise png bytes
  sio = StringIO.StringIO()
  img.save(sio, format="png")
  png_bytes = sio.getvalue()
  
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
  return tf.Summary(value=[tf.Summary.Value(tag="debug_img",
                                            image=tf.Summary.Image(height=img.size[0],
                                                                   width=img.size[1],
                                                                   colorspace=3, # RGB
                                                                   encoded_image_string=png_bytes))])
      
def dice_loss(y, y_hat, batch_size, smoothing=0):
  y = tf.reshape(y, (batch_size, -1))
  y_hat = tf.reshape(y_hat, (batch_size, -1))
  intersection = y * y_hat
  intersection_rs = tf.reduce_sum(intersection, axis=1)
  nom = intersection_rs + smoothing
  denom = tf.reduce_sum(y, axis=1) + tf.reduce_sum(y_hat, axis=1) + smoothing
  score = 2.0 * (nom / denom)
  loss = 1.0 - score
#  loss = tf.Print(loss, [intersection, intersection_rs, nom, denom], first_n=100, summarize=10000)
  return loss
