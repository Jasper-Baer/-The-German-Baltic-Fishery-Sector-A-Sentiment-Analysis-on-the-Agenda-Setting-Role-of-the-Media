# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:10:03 2022

@author: Nutzer
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = r"D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing"
os.chdir(PATH)

from fishery_data_final import load_fishery_data

data = load_fishery_data()
data = data.rename(columns = {'main_articles': 'text'})
data.drop(['level_0', 'index', 'Supporting information (0=no, 1=yes)', '#',
           'Content', 'Autor', 'Heading', 'subdescription', 
           'number of pics', 'content of pics', 'videos (0=no, 1=yes)',
           'number of video', 'content of video', 'Reference',
           'photograph (0=no, 1=yes)', 'Number of SI', 
           'Bermerkung', 'Bemerkung'], axis=1, inplace=True)

data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
data['year'] = data['Date'].dt.year

#data = data[data["Date"] >= "2007-01-01"]
data = data[data["Date"] >= "2009-01-01"]

data['text'].str.replace(';','.')

journal_occur = data.groupby('Journal').size()
newspaper_type = data.groupby('type of newspaper (regional, national)').size()

#journal_occur.to_excel("D:\Studium\PhD\Fischerei\journal_occur.xlsx")
