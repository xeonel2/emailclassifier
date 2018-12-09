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
import sys
import pickle


"""Main Function"""
def mainfunc():
    if len(sys.argv) <= 1:
        print('Please pass the correct argument. (initialize, train,...)')
    elif sys.argv[1] == 'initialize':
        getdataset()
    elif sys.argv[1] == 'train':
        train()
    else:
        print("Invalid Commandline Arguments")

"""Initialization Function"""
def getdataset():
    print('Loading Datasets ham and spam')
    hamfiles = glob.glob("dataset/allham/*.txt")
    spamfiles = glob.glob("dataset/allspam/*.txt")

    allhams = []

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
    
    with open('dataset/training.df', 'wb') as training_file:
        pickle.dump(trainingdf, training_file)
    with open('dataset/test.df', 'wb') as test_file:
        pickle.dump(testdf, test_file)
    
    print("Initialized and parsed datasets to training.df and test.df")
    return

def train():
    print('Training using training.df')
    with open('dataset/training.df', 'rb') as training_file:
        trainingdf = pickle.load(training_file)
    trainingdf.head()
        
mainfunc()