#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:18:46 2018

@author: xeonel
"""
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import re



hamfiles = glob.glob("/home/xeonel/Documents/ee514assignment/dataset/allham/*.txt")
spamfiles = glob.glob("/home/xeonel/Documents/ee514assignment/dataset/allspam/*.txt")

allhams = []

print(hamfiles.count)

for hamfile in hamfiles:
    msg = open(hamfile, encoding="ascii", errors="surrogateescape").read()
    msg = re.findall(r'\w+', msg)
    allhams.append(msg)

allspams = []

for spamfile in spamfiles:
    msg = open(spamfile, encoding="ascii", errors="surrogateescape").read()
    msg = re.findall(r'\w+', msg)
    allspams.append(msg)


hammessagesdf = pd.DataFrame({'content' : pd.Series(allhams),
                           'spam' : False})
    
spammessagesdf = pd.DataFrame({'content' : pd.Series(allspams),
                           'spam' : True})

messagesdf = pd.concat([hammessagesdf, spammessagesdf])

messagesdf = messagesdf.sample(frac=1).reset_index(drop=True)

trainingdf, testdf = train_test_split(messagesdf, test_size=0.3)

""" not touching test set from now xD """