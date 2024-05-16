# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:00:39 2022

@author: Jasper Bär
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the working directory
PATH = r"D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing"
os.chdir(PATH)

# Load the fishery data
from fishery_data_final import load_fishery_data

data = load_fishery_data()

# Ensure 'Date' column is in datetime format and filter data
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
data = data[data["Date"] >= "2009-01-01"]

# Count number of articles per year
year_count = data.groupby(pd.Grouper(key='Date', freq='Y')).Date.count()

# Calculate the monthly average share of articles
month_count = data.groupby(pd.Grouper(key='Date', freq='M')).Date.count()
month_count.index = pd.to_datetime(month_count.index)
monthly_averages = month_count.groupby(month_count.index.month).mean() / 100

# Create a DataFrame for plotting
month_count_prop_all = pd.DataFrame({'proportions': [0] * 12})
month_count_prop_all['proportions'][0:12] = monthly_averages

# Load the quota decision dates
dates = pd.read_excel('D:\Studium\PhD\Fischerei\Raw Data\Complete Data\data, fish & fisheries, SD22-24.xlsx', sheet_name='dates, advice - quota ')['quota decision']
dates = dates.dropna()  # Drop any NaNs
dates = pd.to_datetime(dates)
dates = dates[dates >= "2009-01-01"]

# Generate figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Adjust figsize as needed

# Plot 1: Average yearly share of articles per month
ax1.bar(range(1, 13), month_count_prop_all['proportions'], color='0.4')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax1.set_ylabel("Average yearly share of articles per calendar month", fontsize=14)
ax1.set_title("(a) Average monthly share of articles", y=-0.17, fontsize=16)

# Prepare data for Plot 2

def count_article_per_day(dates,data,n):

    yearly_totals = data.groupby(data['Date'].dt.year).size()
    all_counts = []
    
    for date in dates:
        year = date.year
        year_total = yearly_totals.get(year, 1)  # Use 1 as default to avoid division by zero
        
        mask = (data['Date'] >= date - pd.Timedelta(days=n)) & (data['Date'] <= date + pd.Timedelta(days=n))
        filtered_df = data[mask]
        counts_df = filtered_df.groupby('Date').size()
        reindexed_df = counts_df.reindex(pd.date_range(start=date - pd.Timedelta(days=n), end=date + pd.Timedelta(days=n))).fillna(0)
        fractional_counts = reindexed_df.divide(year_total)
        
        all_counts.append(fractional_counts.values)
        
    return(all_counts)

# Select number of days before and after specified dates
n = 4

all_counts = count_article_per_day(dates,data,n)

all_counts_array = np.array(all_counts)

# Calculate the mean and standard deviation across all counts for each day
means = np.mean(all_counts_array, axis=0)
std_devs = np.std(all_counts_array, axis=0)

# Calculate the standard errors
standard_errors = std_devs / np.sqrt(len(dates))

# Compute the 95% confidence intervals
confidence_intervals = standard_errors * 1.96

# Plot 2: Average yearly share of articles per day around quota dates
days = np.arange(-n, n + 1)
ax2.bar(days, means, yerr=confidence_intervals, capsize=5, color='grey', edgecolor='black')
ax2.set_xlabel('Days relative to date of quota announcement', fontsize=14)
ax2.set_ylabel('Average yearly share of articles per day', fontsize=14)
ax2.set_title("(b) Average daily share of articles around quota announcements", y=-0.17, fontsize=14)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xticks(days)  # Set ticks for every day in the range

plt.tight_layout()  # Adjust layout to make room for all elements
plt.show()

# Generate figure S2

# Load dates for quota advices
dates = pd.read_excel('D:\Studium\PhD\Fischerei\Raw Data\Complete Data\data, fish & fisheries, SD22-24.xlsx', sheet_name = 'dates, advice - quota ')['advice']
dates = dates[:-1] 
dates = [date for date in dates if pd.notna(date) and date != "None\xa0"]

n = 4
all_counts = count_article_per_day(dates,data,n)
    
all_counts_array = np.array(all_counts)

# Calculate the mean and standard deviation across all counts for each day
means = np.mean(all_counts_array, axis=0)
std_devs = np.std(all_counts_array, axis=0)

# Calculate the standard errors
standard_errors = std_devs / np.sqrt(len(dates))

# Compute the 95% confidence intervals
confidence_intervals = standard_errors * 1.96

# Plot 2: Average yearly share of articles per day around quota dates
days = np.arange(-n, n + 1)
plt.bar(days, means, yerr=confidence_intervals, capsize=5, color='grey', edgecolor='black')
plt.set_xlabel('Days relative to date of quota advice', fontsize=14)
plt.set_ylabel('Average yearly share of articles per day', fontsize=14)
plt.set_title("Average daily share of articles around quota advice", y=-0.17, fontsize=14)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.set_xticks(days)  # Set ticks for every day in the range

plt.tight_layout()  # Adjust layout to make room for all elements
plt.show()
