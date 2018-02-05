#!/usr/bin/env python

from __future__ import print_function
import Tkinter as tk
from PIL import Image, ImageTk
import os
import sqlite3
import random
from label_db import LabelDB
import re

class LabelUI():
  def __init__(self, label_db_filename, img_dir):
    
    # what images to review?
    # note: drop trailing / in dir name (if present)
    self.img_dir = re.sub("/$", "", img_dir)  
    self.files = sorted(os.listdir(img_dir))
#    random.shuffle(self.files)
    print("%d files to review" % len(self.files))

    # label db
    self.label_db = LabelDB(label_db_filename)
    self.label_db.create_if_required()

    # TK UI
    root = tk.Tk()
    root.title(label_db_filename)
    root.bind('n', self.display_next_image)
    root.bind('N', self.display_next_unlabelled_image)
    root.bind('p', self.display_previous_image)
    self.canvas = tk.Canvas(root, cursor='tcross')
    self.canvas.config(width=768, height=1024)
    self.canvas.bind('<Button-1>', self.add_bee_event)  # left mouse button
    self.canvas.bind('<Button-3>', self.remove_closest_bee_event)  # right mouse button
    self.canvas.pack()

    # A lookup table from bee x,y to any rectangles that have been drawn
    # in case we want to remove one. the keys of this dict represent all
    # the bee x,y in current image.
    self.x_y_to_boxes = {}  # { (x, y): canvas_id, ... }

    # Main review loop
    self.file_idx = 0
    self.display_new_image()
    root.mainloop()

  def add_bee_event(self, e):
    self.add_bee_at(e.x, e.y)

  def add_bee_at(self, x, y):
    rectangle_id = self.canvas.create_rectangle(x-2,y-2,x+2,y+2, fill='red')
    self.x_y_to_boxes[(x, y)] = rectangle_id
    
  def remove_closest_bee_event(self, e):
    if len(self.x_y_to_boxes) == 0: return
    closest_point = closest_sqr_distance = None
    for x, y in self.x_y_to_boxes.keys():
      sqr_distance = (e.x-x)**2 + (e.y-y)**2
      if sqr_distance < closest_sqr_distance or closest_point is None:
        closest_point = (x, y)
        closest_sqr_distance = sqr_distance
    self.canvas.delete(self.x_y_to_boxes.pop(closest_point))

  def display_next_image(self, e=None):
    self._flush_pending_x_y_to_boxes()
    self.file_idx += 1
    if self.file_idx == len(self.files):
      print("Can't move to image past last image.")
      self.file_idx = len(self.files) - 1      
    self.display_new_image()

  def display_next_unlabelled_image(self, e=None):
    self._flush_pending_x_y_to_boxes()
    while True:
      self.file_idx += 1
      if self.file_idx == len(self.files):
        print("Can't move to image past last image.")
        self.file_idx = len(self.files) - 1
        break
      if not self.label_db.has_labels(self.files[self.file_idx]):
        break
    self.display_new_image()
    
  def display_previous_image(self, e=None):
    self._flush_pending_x_y_to_boxes()
    self.file_idx -= 1
    if self.file_idx < 0:
      print("Can't move to image previous to first image.")
      self.file_idx = 0    
    self.display_new_image()

  def _flush_pending_x_y_to_boxes(self):
    # Flush existing points.
    img_name = self.files[self.file_idx]    
    if len(self.x_y_to_boxes) > 0:
      self.label_db.set_labels(img_name, self.x_y_to_boxes.keys())
      self.x_y_to_boxes.clear()
    
  def display_new_image(self):
    img_name = self.files[self.file_idx]
    # Display image.    
    img = Image.open(self.img_dir + "/" + img_name)
    self.tk_img = ImageTk.PhotoImage(img)
    self.canvas.create_image(0,0, image=self.tk_img, anchor=tk.NW)
    # Look up any existing bees in DB for this image.
    existing_labels = self.label_db.get_labels(img_name)
    for x, y in existing_labels:
      self.add_bee_at(x, y)
    

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--label-db', type=str, required=True)
opts = parser.parse_args() 

LabelUI(opts.label_db, opts.image_dir)
