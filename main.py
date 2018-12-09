#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:18:46 2018

@author: xeonel
"""

import glob
import pandas as pd

hamfiles = glob.glob("/home/xeonel/Documents/ee514assignment/dataset/allham/*.txt")
spamfiles = glob.glob("/home/xeonel/Documents/ee514assignment/dataset/allspam/*.txt")

"""
s = open(hamfiles[0]).read()
print(s)
str(msg, errors='ignore')
"""

allhams = []

print(hamfiles.count)

for hamfile in hamfiles:
    msg = open(hamfile, encoding="ascii", errors="surrogateescape").read()
    allhams.append(msg)

allspams = []

for spamfile in spamfiles:
    msg = open(spamfile, encoding="ascii", errors="surrogateescape").read()
    allspams.append(msg)


hammessagesdf = pd.DataFrame({'content' : pd.Series(allhams),
                           'spam' : False})
    
spammessagesdf = pd.DataFrame({'content' : pd.Series(allspams),
                           'spam' : True})

messagesdf = pd.concat([hammessagesdf, spammessagesdf])

messagesdf = messagesdf.sample(frac=1).reset_index(drop=True)