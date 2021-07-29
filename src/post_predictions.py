#!/usr/bin/env python3

from camtraproc.utils.utils import post_predictions
import pandas as pd
from glob import glob
import os

file_list = [f for f in glob("data/csv/*/*motionm.csv")]

for f in file_list:
    posted = post_predictions(pd.read_csv(f),True)
    if posted:
        os.remove(f)

file_list = [f for f in glob("data/csv/*/*humans.csv")]

for f in file_list:
    posted = post_predictions(pd.read_csv(f),False)
    if posted:
        os.remove(f)

