#!/usr/bin/env python

from __future__ import print_function
import Tkinter as tk
from PIL import Image, ImageTk
import os
import sqlite3
import random

class BeePosDb():
  def __init__(self, directory):
    self.directory = directory
    self.files = os.listdir(directory)
    random.shuffle(self.files)    
    print(len(self.files))

  def next_img(self):
    return self.directory + "/" + self.files.pop()
  
class LabelUI():
  def __init__(self, db):
    self.db = db
    root = tk.Tk()
    root.bind('n', self.display_next_image)
    
    self.canvas = tk.Canvas(root, cursor='tcross')
    self.canvas.config(width=768, height=1024)
    self.canvas.bind('<Button-1>', self.add_bee)  # left mouse button
    self.canvas.bind('<Button-3>', self.remove_closest_bee)  # right mouse button
    self.canvas.pack()

    # A lookup table from bee x,y to any rectangles that have been drawn
    # in case we want to remove one. the keys of this dict represent all
    # the bee x,y
    self.x_y_to_boxes = {}  # { (x, y): canvas_id, ... }
    
    self.display_next_image()
    root.mainloop()

  def add_bee(self, e):
    rectangle_id = self.canvas.create_rectangle(e.x-2,e.y-2,e.x+2,e.y+2, fill='red')
    self.x_y_to_boxes[(e.x, e.y)] = rectangle_id
    
  def remove_closest_bee(self, e):
    if len(self.x_y_to_boxes) == 0: return
    closest_point = closest_sqr_distance = None
    for x, y in self.x_y_to_boxes.keys():
      sqr_distance = (e.x-x)**2 + (e.y-y)**2
      if sqr_distance < closest_sqr_distance or closest_point is None:
        closest_point = (x, y)
        closest_sqr_distance = sqr_distance
    self.canvas.delete(self.x_y_to_boxes.pop(closest_point))

  def display_next_image(self, e=None):
    # Flush existing points.
    if len(self.x_y_to_boxes) > 0:
      print("points!", self.img_name, self.x_y_to_boxes.keys())
      self.x_y_to_boxes.clear()
    # Choose and display next image.
    self.img_name = db.next_img()
    img = Image.open(self.img_name)
    self.tk_img = ImageTk.PhotoImage(img)
    self.canvas.create_image(0,0, image=self.tk_img, anchor=tk.NW)
  

db = BeePosDb("/home/mat/dev/bnn2/images/test3")
LabelUI(db)
