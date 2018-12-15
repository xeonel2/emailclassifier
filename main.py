#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:18:46 2018

@author: Venkatakrishnan Parthasarathy
"""
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import sys
import pickle
import pygal
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


#Main Function
def mainfunc():
    if len(sys.argv) <= 1:
        print('Please pass the correct argument. (initialize, clean, explore, train...)')
    elif sys.argv[1] == 'initialize':
        getdataset()
    elif sys.argv[1] == 'extract':
        extract()
    elif sys.argv[1] == 'explore':
        explore_data()
    else:
        print("Invalid Commandline Arguments")

#Remove Stop words
def stopwordremover(bagofwords):
        returnlist = []
        stopwrds = stopwords.words("english")
        commonwords = ['com', 'please', 'company', '10', 'new', '00', 'may', 'business']
        stopwrds.extend(commonwords)
        for x in bagofwords:
            if x not in stopwrds:
                returnlist.append(x)
        return returnlist

#Dtm
def dtm(messages):
    cv = CountVectorizer()
    fitted = cv.fit_transform(messages)
    df = pd.DataFrame(fitted.toarray(), columns=cv.get_feature_names())
    return cv, df

#Initialization Function
def getdataset():
    print('Loading Datasets ham and spam')
    hamfiles = glob.glob("dataset/allham/*.txt")
    spamfiles = glob.glob("dataset/allspam/*.txt")

    allhams = []

    for hamfile in hamfiles:
        omsg = open(hamfile, encoding="ascii", errors="surrogateescape").read()
        msg = re.findall(r'\w+', omsg)
        msg.pop(0)
        if len(msg) == 0:
            print("message empty, omitting")
            print(omsg)
        else:
            msg = [x.lower() for x in msg]
            msg = stopwordremover(msg)
            allhams.append(msg)


    allspams = []

    for spamfile in spamfiles:
        omsg = open(spamfile, encoding="ascii", errors="surrogateescape").read()
        msg = re.findall(r'\w+', omsg)
        msg.pop(0)
        if len(msg) == 0:
            print("message empty, omitting")
            print(omsg)
        else:
            msg = [x.lower() for x in msg]
            msg = stopwordremover(msg)
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

def extract():
    print('Training using training.df')
    with open('dataset/training.df', 'rb') as training_file:
        trainingdf = pickle.load(training_file)

    smsgsdf = trainingdf[trainingdf['spam'] == True]
    hmsgsdf = trainingdf[trainingdf['spam'] == False]


    smessages = pd.Series(smsgsdf["content"])
    smessages = list(map(lambda x: " ".join(x) , smessages))

    hmessages = pd.Series(hmsgsdf["content"])
    hmessages = list(map(lambda x: " ".join(x) , hmessages))

    #Document term matrix
    spamcv, dtmspam = dtm(smessages)
    hamcv, dtmham = dtm(hmessages)

    spamdist = np.sum(dtmspam, axis=0)
    hamdist = np.sum(dtmham, axis=0)


    #Dumping the word frequencies for visualization and training
    with open('dataset/spamdist.ps', 'wb') as spamdistfile:
        pickle.dump(spamdist, spamdistfile)
    with open('dataset/hamdist.ps', 'wb') as hamdistfile:
        pickle.dump(hamdist, hamdistfile)


    #Getting the length of emails
    smsgcounts = pd.Series(smsgsdf["content"])
    smsgcounts = list(map(lambda x: len(x) , smsgcounts))

    hmsgcounts = pd.Series(hmsgsdf["content"])
    hmsgcounts = list(map(lambda x: len(x) , hmsgcounts))

    #Dumping word length for visualization
    with open('dataset/spamlengths.list', 'wb') as spamlengthsfile:
        pickle.dump(smsgcounts, spamlengthsfile)
    with open('dataset/hamlengths.list', 'wb') as hamlengthsfile:
        pickle.dump(hmsgcounts, hamlengthsfile)

    return


    
def explore_data():    
    with open('dataset/spamdist.ps', 'rb') as spamdistfile:
        spamdist = pickle.load(spamdistfile)
    with open('dataset/hamdist.ps', 'rb') as hamdistfile:
        hamdist = pickle.load(hamdistfile)
    
    #Plot spam frequency
    bar_chart_spam = pygal.HorizontalBar()
    bar_chart_spam.x_labels = spamdist.sort_values(ascending=False)[:20].keys()
    bar_chart_spam.add('spam', spamdist.sort_values(ascending=False)[:20].values)
    bar_chart_spam.render_in_browser()

    #Plot ham frequency
    bar_chart_ham = pygal.HorizontalBar()
    bar_chart_ham.x_labels = hamdist.sort_values(ascending=False)[:20].keys()
    bar_chart_ham.add('ham', hamdist.sort_values(ascending=False)[:20].values)
    bar_chart_ham.render_in_browser()


    with open('dataset/spamlengths.list', 'rb') as spamlengthsfile:
        smsgcounts = pickle.load(spamlengthsfile)
    with open('dataset/hamlengths.list', 'rb') as hamlengthsfile:
        hmsgcounts = pickle.load(hamlengthsfile)

    #plot length of emails
    pyramid_chart = pygal.Pyramid(human_readable=True, legend_at_bottom=True)
    pyramid_chart.title = 'Number of meaningful words'
    pyramid_chart.add('Spam', smsgcounts)
    pyramid_chart.add('Ham', hmsgcounts)
    pyramid_chart.render_in_browser()
    return



mainfunc()