#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:18:46 2018

@author: xeonel
"""

import glob
import pandas as pd

allhams = glob.glob("/home/xeonel/Documents/ee514assignment/dataset/allham/*.txt")
print(allhams[0])

"""
s = open(allhams[0]).read()
print(s)
"""

for hamfile in allhams:
    s = open(hamfile).read()