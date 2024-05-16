# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:19:32 2024

@author: Jasper Bär
"""

import pandas as pd
import os
import pylab as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoLocator, AutoMinorLocator

plt.rcParams['figure.figsize'] = [11, 5]

PATH = r"D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing"
os.chdir(PATH)

from dataprep_v2 import fishery_year, sentiment_index, sentiment_index_quarterly

# Load all sentences
data_sents = pd.read_csv(r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing\fishery_lemmas_sentence_labeled.csv')
data_sents["Date"] = pd.to_datetime(data_sents["Date"], format='%Y-%m-%d')

# Load all sentences from articles related to cod
data_sents_cod = pd.read_csv(r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing\fishery_lemmas_cod_sentences.csv')
data_sents_cod["Date"] = pd.to_datetime(data_sents_cod["Date"], format='%Y-%m-%d')

# Load all sentences from articles related to herring
data_sents_herring = pd.read_csv(r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing\fishery_lemmas_hering_sentences.csv')
data_sents_herring["Date"] = pd.to_datetime(data_sents_herring["Date"], format='%Y-%m-%d')

data_sents_cod = data_sents[data_sents['text'].isin(data_sents_cod['text'])]
data_sents_herring = data_sents[data_sents['text'].isin(data_sents_herring['text'])]

# Load quotes and dates of quota announcments
quota_dates = pd.read_excel('D:\Studium\PhD\Fischerei\Raw Data\Complete Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dates, advice - quota ')['quota decision']
quota_dates = quota_dates[:-1]

quota = pd.read_excel(r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Quotas\advice vs quota, BS fish species_bearbeitet.xlsx', sheet_name = 'dt, quota')
quota = quota[13:-1]
quota["Y"] = pd.to_datetime(quota["Y"], format='%Y')
quota.reset_index(drop = True, inplace = True)

cod_quota = quota.iloc[:,[0,4]]
cod_quota['Unnamed: 4'] = quota.iloc[:,4].pct_change()
cod_quota['quota_abs'] = quota.iloc[:,4]
cod_quota['Y'] = quota_dates

cod_quota = cod_quota[2:]
cod_quota.iloc[:,1] = cod_quota.iloc[:,1]*100

herring_quota = quota.iloc[:,[0,2]]
herring_quota['Unnamed: 2'] = quota.iloc[:,2].pct_change()
herring_quota['quota_abs'] = quota.iloc[:,2]
herring_quota['Y'] = quota_dates

herring_quota = herring_quota[2:]
herring_quota.iloc[:,1] = herring_quota.iloc[:,1]*100

sentiment_all = sentiment_index(fishery_year(data_sents, quota_dates))
sentiment_all['Sentiment_change'] = sentiment_all["Sentiment Index"].pct_change()

sentiment_cod = sentiment_index(fishery_year(data_sents_cod, quota_dates))
sentiment_cod['Sentiment_change'] = sentiment_cod["Sentiment Index"].pct_change()

sentiment_herring = sentiment_index(fishery_year(data_sents_herring, quota_dates))
sentiment_herring['Sentiment_change'] = sentiment_herring["Sentiment Index"].pct_change()

sentiment_all_quarterly = sentiment_index_quarterly(data_sents)
sentiment_cod_quarterly = sentiment_index_quarterly(data_sents_cod)
sentiment_herring_quarterly = sentiment_index_quarterly(data_sents_herring)

sentiment_all['year'] = pd.to_datetime(sentiment_all['year'], format='%Y')
sentiment_cod['year'] = pd.to_datetime(sentiment_cod['year'], format='%Y')
sentiment_herring['year'] = pd.to_datetime(sentiment_herring['year'], format='%Y')

###############################################################################

# Set color for quota and sentiment lines
col_quota = 'black' 
col_sent = 'black' 

###############################################################################
# Generate figure (PLACEHOLDER) 
###############################################################################

fig, ax1 = plt.subplots()

# Plot the rolling mean of the sentiment index
ax1.plot(sentiment_all_quarterly['Date'][15:], sentiment_all_quarterly['Sentiment Index'][15:].rolling(window=3).mean(), color='black', label='Sentiment Index')

# Set the x and y limits and formatting
ax1.set_xlim(datetime(2009, 1, 1), datetime(2022, 1, 1))
ax1.tick_params(axis='y', which='both', left=True, labelleft=True)
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylabel('Sentiment', fontsize=24)
ax1.set_xticks(ax1.get_xticks()[1:-1])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.yaxis.set_major_locator(MultipleLocator(0.05))
ax1.set_ylim(-0.4, -0.1)

# Updated dictionary for abbreviations
annotations = {
    'A': 'Cod stocks doubled',
    'B': 'Strong herring quota decrease',
    'C': 'High cod stocks',
    'D': 'Recovering fish stocks',
    'E': 'Herring quota increase',
    'F': 'Bad year for cod offspring',
    'G': 'Strong cod and herring quota decrease',
    'H': 'Fishing ban in cod spawning areas',
    'I': 'Strong cod quota and stock increase',
    'J': 'Baltic Sea Herring loses MSC certification',
    'K': 'Cod stocks almost at tipping point',
}

# Updated list of important dates with new labels
important_dates = [
    (datetime(2009, 10, 1), 'A'),
    (datetime(2010, 7, 1), 'B'),
    (datetime(2011, 4, 1), 'C'),
    (datetime(2012, 7, 1), 'D'),
    (datetime(2013, 7, 1), 'D'),
    (datetime(2014, 10, 1), 'E'),
    (datetime(2015, 10, 1), 'F'),
    (datetime(2016, 7, 1), 'G'),
    (datetime(2017, 1, 1), 'H'),
    (datetime(2018, 1, 1), 'I'),
    (datetime(2018, 10, 1), 'J'),
    (datetime(2019, 7, 1), 'K'),
    (datetime(2019, 10, 1), 'I'),
]

for date, label in important_dates:
    date_index = sentiment_all_quarterly[sentiment_all_quarterly.iloc[:,2] == date].index
    important_value = sentiment_all_quarterly['Sentiment Index'][4:].rolling(window=3).mean()[date_index]
    
    ax1.plot(date, important_value, 'bo', markersize=10)  
    ax1.annotate(label, xy=(date, important_value), xytext=(date, important_value - 0.02),
                 textcoords='data', fontsize=14, ha='center', va='top')

# Custom legend in multiple rows
handles = [plt.Line2D([0], [0], color='w', label=f'{abbr}: {full_text}')
           for abbr, full_text in annotations.items()]
ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=3, fontsize=10)

plt.show()

###############################################################################
# Generate figure (PLACEHOLDER) 
###############################################################################

fig, (ax1,ax3) = plt.subplots(1,2, figsize=(27.5,6)) 
 
ax2 = ax1.twinx() 
ax1.plot(quota_dates[1:], cod_quota.iloc[:,1], color = col_quota, linestyle = '--') 
ax2.plot(sentiment_cod_quarterly['Date'][7:], sentiment_cod_quarterly['Sentiment Index'][7:].rolling(window=3).mean(), color = col_sent)
ax1.hlines(y=0, color='black', linestyle='-', xmin = quota_dates[2:].iloc[0], xmax = quota_dates.iloc[-1]) 
 
ax1.set_xlim(datetime(2006, 9, 1), datetime(2021, 12, 31))

ax1.tick_params(axis='y', colors=col_quota, labelsize = 18) 
ax2.tick_params(axis='y', colors=col_sent, labelsize = 18) 
 
ax1.tick_params(axis='x', labelsize = 18) 
 
ax1.set_ylabel('Cod quota (% change)', fontsize = 24) 
ax2.set_ylabel('Sentiment', fontsize = 24) 
 
ax1.set_title('(a) Cod', y=-0.25, fontsize = 24) 
 
ax1.set_xticks(ax1.get_xticks()[1:-1]) 
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45) 
 
ax1.yaxis.label.set_color(col_quota) 
ax2.yaxis.label.set_color(col_sent) 
 
ax1.xaxis.set_major_locator(mdates.YearLocator()) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 

ax1.set_ylim(-100, 80)
ax2.set_ylim(-0.5, -0.1)
 
###

ax4 = ax3.twinx()
ax3.plot(quota_dates[1:], herring_quota.iloc[:,1], color = col_quota, linestyle='--') 
ax4.plot(sentiment_herring_quarterly['Date'][2:], sentiment_herring_quarterly['Sentiment Index'][2:].rolling(window=3).mean(), color = col_sent) 
ax3.hlines(y=0, color='black', xmin = quota_dates[2:].iloc[0], xmax = quota_dates.iloc[-1]) 

ax3.set_xlim(datetime(2006, 9, 1), datetime(2021, 12, 31))

ax3.tick_params(axis='y', colors=col_quota, labelsize = 18) 
ax4.tick_params(axis='y', colors=col_sent, labelsize = 18) 
 
ax3.tick_params(axis='x', labelsize = 18) 
 
ax4.set_ylabel('Sentiment', fontsize = 24) 
ax3.set_ylabel( 'Herring quota (% change)', fontsize = 24) 
 
ax3.set_title('(b) Herring', y=-0.25, fontsize = 24) 
 
ax3.set_xticks(ax3.get_xticks()[1:-1]) 
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45) 
 
ax3.yaxis.label.set_color(col_quota) 
ax4.yaxis.label.set_color(col_sent) 
 
ax3.xaxis.set_major_locator(mdates.YearLocator()) 
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 

ylim1_ax1 = ax1.get_ylim()
ylim1_ax2 = ax2.get_ylim()
ylim1_ax3 = ax3.get_ylim()
ylim1_ax4 = ax4.get_ylim()

ax3.set_ylim(-100, 80)
ax4.set_ylim(-0.5, -0.1)
 
plt.subplots_adjust(wspace=0.4, hspace=0) 
plt.show() 