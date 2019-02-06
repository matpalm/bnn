#!/usr/bin/env python3

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import sqlite3
import random
from label_db import LabelDB
import re

class LabelUI():
  def __init__(self, label_db_filename, img_dir, width, height, sort=True):

    # what images to review?
    # note: drop trailing / in dir name (if present)
    self.img_dir = re.sub("/$", "", img_dir)
    self.files = os.listdir(img_dir)
    if sort:
      self.files = sorted(self.files)
    else:
      random.shuffle(self.files)
    print("%d files to review" % len(self.files))

    # label db
    self.label_db = LabelDB(label_db_filename)
    self.label_db.create_if_required()

    # TK UI
    root = tk.Tk()
    root.title(label_db_filename)
    root.bind('<Right>', self.display_next_image)
    print("RIGHT  next image")
    root.bind('<Left>', self.display_previous_image)
    print("LEFT   previous image")
    root.bind('<Up>', self.toggle_bees)
    print("UP     toggle labels")
    root.bind('N', self.display_next_unlabelled_image)
    print("N   next image with 0 labels")
    root.bind('Q', self.quit)
    print("Q   quit")
    self.canvas = tk.Canvas(root, cursor='tcross')
    self.canvas.config(width=width, height=height)
    self.canvas.bind('<Button-1>', self.add_bee_event)  # left mouse button
    self.canvas.bind('<Button-3>', self.remove_closest_bee_event)  # right mouse button

    self.canvas.pack()

    # A lookup table from bee x,y to any rectangles that have been drawn
    # in case we want to remove one. the keys of this dict represent all
    # the bee x,y in current image.
    self.x_y_to_boxes = {}  # { (x, y): canvas_id, ... }

    # a flag to denote if bees are being displayed or not
    # while no displayed we lock down all img navigation
    self.bees_on = True

    # Main review loop
    self.file_idx = 0
    self.display_new_image()
    root.mainloop()

  def quit(self, e):
        exit()

  def add_bee_event(self, e):
    if not self.bees_on:
      print("ignore add bee; bees not on")
      return
    self.add_bee_at(e.x, e.y)

  def add_bee_at(self, x, y):
    rectangle_id = self.canvas.create_rectangle(x-2,y-2,x+2,y+2, fill='red')
    self.x_y_to_boxes[(x, y)] = rectangle_id

  def remove_bee(self, rectangle_id):
    self.canvas.delete(rectangle_id)

  def toggle_bees(self, e):
    if self.bees_on:
      # store x,y s in tmp list and delete all rectangles from canvas
      self.tmp_x_y = []
      for (x, y), rectangle_id in self.x_y_to_boxes.items():
        self.remove_bee(rectangle_id)
        self.tmp_x_y.append((x, y))
      self.x_y_to_boxes = {}
      self.bees_on = False
    else:  # bees not on
      # restore all temp stored bees
      for x, y in self.tmp_x_y:
        self.add_bee_at(x, y)
      self.bees_on = True

  def remove_closest_bee_event(self, e):
    if not self.bees_on:
      print("ignore remove bee; bees not on")
      return
    if len(self.x_y_to_boxes) == 0: return
    closest_point = None
    closest_sqr_distance = 0.0
    for x, y in self.x_y_to_boxes.keys():
      sqr_distance = (e.x-x)**2 + (e.y-y)**2
      if sqr_distance < closest_sqr_distance or closest_point is None:
        closest_point = (x, y)
        closest_sqr_distance = sqr_distance
    self.remove_bee(self.x_y_to_boxes.pop(closest_point))

  def display_next_image(self, e=None):
    if not self.bees_on:
      print("ignore move to next image; bees not on")
      return
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
    if not self.bees_on:
      print("ignore move to previous image; bees not on")
      return
    self._flush_pending_x_y_to_boxes()
    self.file_idx -= 1
    if self.file_idx < 0:
      print("Can't move to image previous to first image.")
      self.file_idx = 0
    self.display_new_image()

  def _flush_pending_x_y_to_boxes(self):
    # Flush existing points.
    img_name = self.files[self.file_idx]
    self.label_db.set_labels(img_name, self.x_y_to_boxes.keys())
    self.x_y_to_boxes.clear()

  def display_new_image(self):
    img_name = self.files[self.file_idx]
    # Display image (with filename added)
    title = img_name + " " + str(self.file_idx) + " of " + str(len(self.files)-1)
    img = Image.open(self.img_dir + "/" + img_name)
    canvas = ImageDraw.Draw(img)
    canvas.text((0,0), title, fill='black')
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
parser.add_argument('--width', type=int, default=768, help='input image width')
parser.add_argument('--height', type=int, default=1024, help='input image height')
parser.add_argument('--no-sort', action='store_true')
opts = parser.parse_args()

LabelUI(opts.label_db, opts.image_dir, opts.width, opts.height, sort=not opts.no_sort)
