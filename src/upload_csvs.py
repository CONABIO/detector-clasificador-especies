#!/usr/bin/env python3

from camtraproc.utils.utils import upload_s3
from glob import glob
import os

x1 = [f for f in glob("data/results/*/*results.csv")]
x2 = [f for f in glob("data/csv/*/*.csv")  if 'files_coor.csv' not in f]
x3 = [f for f in glob("data/results/*/*results_no_coor.csv")]
file_list = x1 + x2 + x3
file_list

for f in file_list:
    uploaded = upload_s3('snmb2','',f,region='us-west-2')
    if uploaded:
        os.remove(f)

