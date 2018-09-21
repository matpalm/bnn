import sqlite3
import json

class LabelDB(object):
  def __init__(self, label_db_file='label.db', check_same_thread=True):
    self.conn = sqlite3.connect(label_db_file, check_same_thread=check_same_thread)

  def create_if_required(self):
    # called once to create db
      c = self.conn.cursor()
      try:
        c.execute('''create table imgs (
                          id integer primary key autoincrement,
                          filename text
                     )''')
        c.execute('''create table labels (
                          img_id integer,
                          x integer,
                          y integer
                     )''')
      except sqlite3.OperationalError:
        # assume table already exists? clumsy...
        pass

  def has_been_created(self):
    c = self.conn.cursor()
    c.execute("select name from sqlite_master where type='table' AND name='imgs';")
    return c.fetchone() is not None

  def imgs(self):
    c = self.conn.cursor()
    c.execute("select filename from imgs")
    return set(map(lambda f: f[0], c.fetchall()))

  def has_labels(self, img):
    id = self._id_for_img(img)
    return id is not None

  def get_labels(self, img):
    if not self.has_labels(img):
      return []
    c = self.conn.cursor()
    c.execute("""select l.x, l.y
                 from labels l join imgs i on l.img_id = i.id
                 where i.filename=?""", (img,))
    return c.fetchall()

  def set_labels(self, img, labels, flip=False):
    img_id = self._id_for_img(img)
    if img_id is None:
      img_id = self._create_row_for_img(img)
    else:
      self._delete_labels_for_img_id(img_id)
    self._add_rows_for_labels(img_id, labels, flip=flip)

  def _id_for_img(self, img):
    c = self.conn.cursor()
    c.execute("select id from imgs where filename=?", (img,))
    id = c.fetchone()
    if id is None:
      return None
    else:
      return id[0]

  def _create_row_for_img(self, img):
    c = self.conn.cursor()
    c.execute("insert into imgs (filename) values (?)", (img,))
    self.conn.commit()
    return self._id_for_img(img)

  def _delete_labels_for_img_id(self, img_id):
    c = self.conn.cursor()
    c.execute("delete from labels where img_id=?", (img_id,))
    self.conn.commit()

  def _add_rows_for_labels(self, img_id, labels, flip=False):
    c = self.conn.cursor()
    for x, y in labels:
      if flip:
        # TODO: DANGER WILL ROBERTSON! the existence of this, for the population
        #       of db from centroids_of_connected_components denotes some inconsistency
        #       somewhere... :/
        x, y = y, x
      c.execute("insert into labels (img_id, x, y) values (?, ?, ?)", (img_id, x, y,))
    self.conn.commit()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--label-db', type=str, default="label.db")
  opts = parser.parse_args()
  db = LabelDB(label_db_file=opts.label_db)
  print("\n".join(db.imgs()))
