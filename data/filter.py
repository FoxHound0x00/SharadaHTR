import os
import shutil
import re

DIR = "output/"
DST_DIR = "temp_annot/"

os.makedirs(DST_DIR,exist_ok=True)
files = os.listdir(DIR)


for f in files:
    if f.endswith('.json'):
        json_file = os.path.join(DIR,f)
        img_file = os.path.join(DIR,os.path.splitext(f)[0]+'.png')
        if os.path.isfile(img_file):
            shutil.copy(img_file,DST_DIR)
            shutil.copy(json_file,DST_DIR)