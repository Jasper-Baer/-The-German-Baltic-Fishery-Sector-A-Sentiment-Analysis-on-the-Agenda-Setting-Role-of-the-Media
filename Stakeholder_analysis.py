# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:55:22 2023

@author: Jasper Bär
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.dates as mdates
from Stakeholder_analysis_util import stakeholder_analysis 

from openpyxl import Workbook

plt.rcParams['figure.figsize'] = [11, 5]

data_path = r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Preprocessing\fishery_lemmas_sentence_labeled.csv'
pers_path = r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Topic and Stakeholder Analysis\common_pers_cleaned_170223.xlsx'
orgs_path = r'D:\Studium\PhD\Github\Fischerei - Master\Sentiment-Analysis-Western-Baltic-Sea\Topic and Stakeholder Analysis\common_org_cleaned_170223.xlsx'
quota_path = r'D:\Studium\PhD\Fischerei\Raw Data\Complete Data\data, fish & fisheries, SD22-24.xlsx'
start_year = 2009
end_year = 2022

def load_and_preprocess_data(data_path, pers_path, orgs_path, quota_path):
    # Specify data types for columns
    dtype_dict = {
        'Unnamed: 0.1': 'int64',
        'Unnamed: 0': 'int64',
        'id': 'str',
        'year': 'int64',
        'Journal': 'str',
        'type of newspaper (regional, national)': 'str',
        'Date': 'str',  
        'Category': 'str',
        'preheading': 'str',
        'Klima': 'float64',
        'Naturschutz': 'float64',
        'Fischerei': 'float64',
        'Unnamed: 23': 'str',
        'lemmas': 'str',
        'text': 'str',
        'word_count': 'int64',
        'Label': 'int64'
    }

    # Load data with specified dtypes
    data = pd.read_csv(data_path, dtype=dtype_dict, low_memory=False)
    data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')
    data = data.rename(columns={'Lemmas': 'lemmas'})
    data = data.dropna(subset=['lemmas'])

    # Load and transform lists with organisations and names
    common_pers = pd.read_excel(pers_path)
    common_orgs = pd.read_excel(orgs_path)
    
    common_orgs.dropna(subset=['Name'], inplace=True)
    common_orgs['Lemma1'] = [common_orgs['Name'].iloc[idx].lower() if name != name else name for idx, name in enumerate(common_orgs['Lemma1'])]
    common_orgs['Abkürzung'] = [common_orgs['Abkürzung'].iloc[idx].lower() if name == name else float('nan') for idx, name in enumerate(common_orgs['Abkürzung'])]
    
    quota_dates = pd.read_excel(quota_path, sheet_name='dates, advice - quota ')['quota decision']
    quota_dates = quota_dates[(quota_dates.dt.year >= start_year) & (quota_dates.dt.year <= end_year)][:-1].reset_index(drop=True)
    
    common_pers['Name'] = [nam.lower() for nam in common_pers['Name']]
    common_pers['Name'] = common_pers['Name'].str.replace("von ", "", regex=False)
    
    # Select all stakeholders
    stakeholders_pers_list = list(common_pers[common_pers['Stakeholder-Gruppe'].str.contains('politics|management|science|fisheries|engo|rf', na=False)]['Name'])
    stakeholders_orgs_list = list(common_orgs[common_orgs['Stakeholder-Gruppe'].str.contains('politics|management|science|fisheries|engo|rf', na=False)]['Name'])

    return data, common_pers, common_orgs, quota_dates, stakeholders_pers_list, stakeholders_orgs_list

def run_stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates):
    return stakeholder_analysis(
        data, 
        common_pers, 
        common_orgs, 
        stakeholders_pers_list, 
        stakeholders_orgs_list, 
        start_year, 
        end_year, 
        quota_dates
    )

def analyze_by_group(common_pers, common_orgs, group_name, data, start_year, end_year, quota_dates):
    stakeholders_pers_list = list(common_pers[common_pers['Stakeholder-Gruppe'].str.contains(group_name, na=False)]['Name'])
    stakeholders_orgs_list = list(common_orgs[common_orgs['Stakeholder-Gruppe'].str.contains(group_name, na=False)]['Name'])
    return run_stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates)

def analyze_by_governance_level(common_pers, common_orgs, level, data, start_year, end_year, quota_dates):
    if level == 'supranational|international':
        stakeholders_pers_list = list(common_pers[common_pers['Governance level'].str.contains(level, na=False)]['Name'])
        stakeholders_orgs_list = list(common_orgs[common_orgs['Governance level'].str.contains(level, na=False)]['Name'])
    else:
        stakeholders_pers_list = list(common_pers[common_pers['Governance level'] == level]['Name'])
        stakeholders_orgs_list = list(common_orgs[common_orgs['Governance level'] == level]['Name'])
    return run_stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates)

def calculate_summary_stats(data):
    return data.describe()[['mean', 'std', 'min', 'max']]

def write_summary_to_excel(summary_stats, filename):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary Statistics"
    
    # Write header for Table S6 (Sentiment by groups)
    ws.append(["Table S6. Stakeholder sentiment by groups"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly sentiment politics & authorities individuals", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_persons_sentiment'].round(3)
        ws.append([f"Yearly sentiment {group.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly sentiment Politics & Authorities Organizations", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_organizations_sentiment'].round(3)
        ws.append([f"Yearly sentiment {group.replace('|', ' & ')} organizations"] + list(stats.values))
    
    ws.append([])
    ws.append(["Both"])
    ws.append(["Yearly sentiment Politics & Authorities Both", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_both_sentiment'].round(3)
        ws.append([f"Yearly sentiment {group.replace('|', ' & ')} both"] + list(stats.values))

    # Write header for Table S7 (Sentiment by governance level)
    ws.append([])
    ws.append(["Table S7. Stakeholder sentiment by governance level"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly sentiment subnational individuals", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_persons_sentiment'].round(3)
        ws.append([f"Yearly sentiment {level.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly sentiment subnational organizations", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_organizations_sentiment'].round(3)
        ws.append([f"Yearly sentiment {level.replace('|', ' & ')} organizations"] + list(stats.values))

    ws.append([])
    ws.append(["Both"])
    ws.append(["Yearly sentiment subnational both", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_both_sentiment'].round(3)
        ws.append([f"Yearly sentiment {level.replace('|', ' & ')} both"] + list(stats.values))

    # Write header for Table S8 (Shares by groups)
    ws.append([])
    ws.append(["Table S8. Stakeholder shares by groups (multiple occurrences of same stakeholder per articles)"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly share politics & authorities individuals", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_persons_share'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly share Politics & Authorities Organizations", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_organizations_share'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} organizations"] + list(stats.values))
    
    ws.append([])
    ws.append(["Both"])
    ws.append(["Yearly share Politics & Authorities Both", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_both_share'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} both"] + list(stats.values))

    # Write header for Table S9 (Shares by governance level)
    ws.append([])
    ws.append(["Table S9. Stakeholder shares by governance level (multiple occurrences of same stakeholder per articles)"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly share subnational individuals", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_persons_share'].round(3)
        ws.append([f"Yearly share {level.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly share subnational organizations", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_organizations_share'].round(3)
        ws.append([f"Yearly share {level.replace('|', ' & ')} organizations"] + list(stats.values))

    # Write header for Table S10 (Shares by groups, counting a stakeholder only once per article)
    ws.append([])
    ws.append(["Table S10. Stakeholder shares by groups (counting a stakeholder only once per article)"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly share politics & authorities individuals", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_persons_share_once'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly share Politics & Authorities Organizations", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_organizations_share_once'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} organizations"] + list(stats.values))
    
    ws.append([])
    ws.append(["Both"])
    ws.append(["Yearly share Politics & Authorities Both", "Mean", "S.D.", "Min", "Max"])
    for group in ['politics|management', 'science', 'fisheries', 'engo']:
        stats = summary_stats[f'{group}_both_share_once'].round(3)
        ws.append([f"Yearly share {group.replace('|', ' & ')} both"] + list(stats.values))

    # Write header for Table S11 (Shares by governance level, counting a stakeholder only once per article)
    ws.append([])
    ws.append(["Table S11. Stakeholder shares by governance level (counting a stakeholder only once per article)"])
    ws.append([])
    ws.append(["Individuals"])
    ws.append(["Yearly share subnational individuals", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_persons_share_once'].round(3)
        ws.append([f"Yearly share {level.replace('|', ' & ')} individuals"] + list(stats.values))
    
    ws.append([])
    ws.append(["Organizations"])
    ws.append(["Yearly share subnational organizations", "Mean", "S.D.", "Min", "Max"])
    for level in ['local', 'national', 'supranational|international']:
        stats = summary_stats[f'{level}_organizations_share_once'].round(3)
        ws.append([f"Yearly share {level.replace('|', ' & ')} organizations"] + list(stats.values))

    wb.save(filename)
    
def main(data_path, pers_path, orgs_path, quota_path, start_year, end_year):
    data, common_pers, common_orgs, quota_dates, stakeholders_pers_list, stakeholders_orgs_list = load_and_preprocess_data(data_path, pers_path, orgs_path, quota_path)
    
    # Initial stakeholder analysis
    results = run_stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates)
    stakeholders_pers_sum_all_persons, stakeholders_orgs_sum_all_orgs = results[9], results[10]
    
    # Filter stakeholders with a minimum value
    stakeholders_orgs_sum_all_orgs = stakeholders_orgs_sum_all_orgs[stakeholders_orgs_sum_all_orgs['Value'] >= 20].copy()
    stakeholders_pers_sum_all_persons = stakeholders_pers_sum_all_persons[stakeholders_pers_sum_all_persons['Value'] >= 20].copy()

    stakeholders_pers_sum_all_persons.reset_index(inplace=True)
    stakeholders_pers_sum_all_persons['Name'] = stakeholders_pers_sum_all_persons['Name'].str.lower().str.strip()
    common_pers['Name'] = common_pers['Name'].str.lower().str.strip()

    common_pers = pd.merge(common_pers, stakeholders_pers_sum_all_persons, on='Name', how='inner')

    stakeholders_orgs_sum_all_orgs.reset_index(inplace=True)
    stakeholders_orgs_sum_all_orgs['Name'] = stakeholders_orgs_sum_all_orgs['Name'].str.lower().str.strip()
    common_orgs['Name'] = common_orgs['Name'].str.lower().str.strip()

    common_orgs = pd.merge(common_orgs, stakeholders_orgs_sum_all_orgs, on='Name', how='inner')

    stakeholders_pers_list = list(common_pers[common_pers['Stakeholder-Gruppe'].str.contains('politics|management|science|fisheries|engo', na=False)]['Name'])
    stakeholders_orgs_list = list(common_orgs[common_orgs['Stakeholder-Gruppe'].str.contains('politics|management|science|fisheries|engo', na=False)]['Name'])

    # Run stakeholder analysis again with filtered data
    results = run_stakeholder_analysis(data, common_pers, common_orgs, stakeholders_pers_list, stakeholders_orgs_list, start_year, end_year, quota_dates)
    
    # Further analysis based on stakeholder groups
    groups = ['politics|management', 'science', 'fisheries', 'engo']
    analysis_results = {group: analyze_by_group(common_pers, common_orgs, group, data, start_year, end_year, quota_dates) for group in groups}
    
    # Further analysis based on governance levels
    governance_levels = ['local', 'national', 'supranational|international']
    governance_results = {level: analyze_by_governance_level(common_pers, common_orgs, level, data, start_year, end_year, quota_dates) for level in governance_levels}

    # Summary statistics for each group and governance level
    summary_stats = {}

    for group, result in analysis_results.items():
        stakeholders_pers_sum = result[0]
        sentiment_pers_sum = result[1]
        
        stakeholders_orgs_sum = result[2]
        sentiment_orgs_sum = result[3]
        
        stakeholders_both_sum = result[4]
        sentiment_both_sum = result[5]
        
        summary_stats[f'{group}_persons_sentiment'] = calculate_summary_stats(sentiment_pers_sum['Sentiment Index'])
        summary_stats[f'{group}_organizations_sentiment'] = calculate_summary_stats(sentiment_orgs_sum['Sentiment Index'])
        summary_stats[f'{group}_both_sentiment'] = calculate_summary_stats(sentiment_both_sum['Sentiment Index'])
        
        summary_stats[f'{group}_persons_share'] = calculate_summary_stats(stakeholders_pers_sum['share'])
        summary_stats[f'{group}_organizations_share'] = calculate_summary_stats(stakeholders_orgs_sum['share'])
        summary_stats[f'{group}_both_share'] = calculate_summary_stats(stakeholders_both_sum['share'])
        
        summary_stats[f'{group}_persons_share_once'] = calculate_summary_stats(stakeholders_pers_sum['share_once'])
        summary_stats[f'{group}_organizations_share_once'] = calculate_summary_stats(stakeholders_orgs_sum['share_once'])
        summary_stats[f'{group}_both_share_once'] = calculate_summary_stats(stakeholders_both_sum['share_once'])
    
    for level, result in governance_results.items():
        stakeholders_pers_sum = result[0]
        sentiment_pers_sum = result[1]
        
        stakeholders_orgs_sum = result[2]
        sentiment_orgs_sum = result[3]
        
        stakeholders_both_sum = result[4]
        sentiment_both_sum = result[5]
        
        summary_stats[f'{level}_persons_sentiment'] = calculate_summary_stats(sentiment_pers_sum['Sentiment Index'])
        summary_stats[f'{level}_organizations_sentiment'] = calculate_summary_stats(sentiment_orgs_sum['Sentiment Index'])
        summary_stats[f'{level}_both_sentiment'] = calculate_summary_stats(sentiment_both_sum['Sentiment Index'])
        
        summary_stats[f'{level}_persons_share'] = calculate_summary_stats(stakeholders_pers_sum['share'])
        summary_stats[f'{level}_organizations_share'] = calculate_summary_stats(stakeholders_orgs_sum['share'])
        summary_stats[f'{level}_both_share'] = calculate_summary_stats(stakeholders_both_sum['share'])
        
        summary_stats[f'{level}_persons_share_once'] = calculate_summary_stats(stakeholders_pers_sum['share_once'])
        summary_stats[f'{level}_organizations_share_once'] = calculate_summary_stats(stakeholders_orgs_sum['share_once'])
        summary_stats[f'{level}_both_share_once'] = calculate_summary_stats(stakeholders_both_sum['share_once'])
    

    for group, result in analysis_results.items():
        stakeholders_both_sum = result[4]
        stakeholders_both_sum.index = pd.to_datetime(stakeholders_both_sum.index, format='%Y')
    
    for level, result in governance_results.items():
        stakeholders_both_sum = result[4]
        stakeholders_both_sum.index = pd.to_datetime(stakeholders_both_sum.index, format='%Y')

    return results, analysis_results, governance_results, quota_dates, summary_stats

if __name__ == "__main__":
    results, analysis_results, governance_results, quota_dates, summary_stats = main(data_path, pers_path, orgs_path, quota_path, start_year, end_year)
        
    stakeholders_both_sum_pm = analysis_results['politics|management'][4]
    stakeholders_both_sum_f = analysis_results['fisheries'][4]
    stakeholders_both_sum_s = analysis_results['science'][4]
    stakeholders_both_sum_e = analysis_results['engo'][4]

    stakeholders_sentiment_both_pm = analysis_results['politics|management'][5]
    stakeholders_sentiment_both_f = analysis_results['fisheries'][5]
    stakeholders_sentiment_both_s = analysis_results['science'][5]
    stakeholders_sentiment_both_e = analysis_results['engo'][5]
    
    stakeholders_both_sum_local = governance_results['local'][4]
    stakeholders_both_sum_national = governance_results['national'][4]
    stakeholders_both_sum_supranational = governance_results['supranational|international'][4]

    stakeholders_sentiment_both_local = governance_results['local'][5]
    stakeholders_sentiment_both_national = governance_results['national'][5]
    stakeholders_sentiment_both_supranational = governance_results['supranational|international'][5]
    
    years = quota_dates[:-1].dt.to_period('Y').dt.to_timestamp()

    ###############################################################################
    
    # Generate figure 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=400)
    
    color1 = (0, 89, 84) 
    color2 = (254, 217, 145)
    color3 = (99, 194, 203) 
    color4 = (244,177,131) 
    
    # Convert RGB to matplotlib color format
    def convert_color(rgb):
        return tuple([x / 255. for x in rgb])
    
    color1 = convert_color(color1)
    color2 = convert_color(color2)
    color3 = convert_color(color3)
    color4 = convert_color(color4)
    
    wi = 200
    
    # First graph (Share)
    total = stakeholders_both_sum_pm['share'] + stakeholders_both_sum_f['share'] + stakeholders_both_sum_s['share'] + stakeholders_both_sum_e['share']
    
    ax1.bar(years, stakeholders_both_sum_pm['share'] / total * 100, color=color1, label='Politics & Public Authorities', width=wi)
    ax1.bar(years, stakeholders_both_sum_f['share'] / total * 100, bottom=stakeholders_both_sum_pm['share'] / total * 100, color=color2, label='Fishery', width=wi)
    ax1.bar(years, stakeholders_both_sum_s['share'] / total * 100, bottom=(stakeholders_both_sum_pm['share'] + stakeholders_both_sum_f['share']) / total * 100, color=color3, label='Science', width=wi)
    ax1.bar(years, stakeholders_both_sum_e['share'] / total * 100, bottom=(stakeholders_both_sum_pm['share'] + stakeholders_both_sum_f['share'] + stakeholders_both_sum_s['share']) / total * 100, color=color4, label='eNGO', width=wi)
    
    ax1.tick_params(axis='y', labelsize = 14)
    
    ax1.set_xticks(ax1.get_xticks()[1:-1])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    ax1.set_xlim(datetime.datetime(2008, 6, 1), datetime.datetime(2020, 12, 30))
    
    ax1.tick_params(axis='x', labelsize = 14)
    
    ax1.set_ylabel('Share', fontsize = 22)
    
    ax1.set_title('(a) Stakeholder share', y=-0.33, fontsize = 22)
    
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Second graph (Sentiment)
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_pm['Sentiment Index']), color=color1, linewidth=3, linestyle='-', label='Politics & management')
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_f['Sentiment Index']), color=color2, linewidth=3, linestyle='-', label='Fishery')
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_s['Sentiment Index']), color=color3, linewidth=3, linestyle='-', label='Science')
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_e['Sentiment Index']), color=color4, linewidth=3, linestyle='-', label='eNGO') 
    
    ax2.tick_params(axis='y', labelsize = 14)
    
    ax2.set_ylabel('Sentiment', fontsize = 22)
    
    ax2.set_xlim(datetime.datetime(2009, 1, 1), datetime.datetime(2021, 12, 31))
    
    ax2.set_title('(b) Stakeholder sentiment', y=-0.33, fontsize = 22)
    
    ax2.set_xticks(ax2.get_xticks()[1:-1])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    ax2.tick_params(axis='x', labelsize = 14)
    
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax2.set_ylim(-0.75, -0.05)
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), shadow=False, ncol=4, fontsize=18, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 1]) 
    
    plt.show()

    ###############################################################################
    
    # Generate figure S5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=400)
    
    color1 = (0, 60, 71) 
    color2 = (0, 173, 189)
    color3 = (204, 200, 192)  
    
    def convert_color(rgb):
        return tuple([x / 255. for x in rgb])
    
    color1 = convert_color(color1)
    color2 = convert_color(color2)
    color3 = convert_color(color3)
    
    wi = 200
    
    total = stakeholders_both_sum_local['share'] + stakeholders_both_sum_national['share'] + stakeholders_both_sum_supranational['share']
    
    # First graph (Share)
    ax1.bar(years, stakeholders_both_sum_local['share']/ total * 100, color=color1, label='Subnational', width=wi)
    ax1.bar(years, stakeholders_both_sum_national['share']/ total * 100, bottom=stakeholders_both_sum_local['share']/ total * 100, color=color2, label='National', width=wi)
    ax1.bar(years, stakeholders_both_sum_supranational['share']/ total * 100, bottom=stakeholders_both_sum_local['share']/ total * 100 + stakeholders_both_sum_national['share']/ total * 100, color=color3, label='International', width=wi)
            
    ax1.tick_params(axis='y', labelsize = 14)
    
    ax1.set_xticks(ax1.get_xticks()[1:-1])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    ax1.set_xlim(datetime.datetime(2008, 6, 1), datetime.datetime(2020, 6, 30))
    
    ax1.tick_params(axis='x', labelsize = 14)
    
    ax1.set_ylabel('Share', fontsize = 22)
    
    ax1.set_title('(a) Stakeholder share', y=-0.33, fontsize = 22)
    
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Second graph (Sentiment)
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_local['Sentiment Index']), color=color1, linewidth=3, linestyle='-', label='Local')
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_national['Sentiment Index']), color=color2, linewidth=3, linestyle='-', label='National')
    ax2.plot(quota_dates[:-1], np.array(stakeholders_sentiment_both_supranational['Sentiment Index']), color=color3, linewidth=3, linestyle='-', label='International')
      
    ax2.tick_params(axis='y', labelsize = 14)
    
    ax2.set_ylabel('Sentiment', fontsize = 22)
    
    ax2.set_xlim(datetime.datetime(2009, 1, 1), datetime.datetime(2021, 12, 31))
    
    ax2.set_title('(b) Stakeholder sentiment', y=-0.33, fontsize = 22)
    
    ax2.set_xticks(ax2.get_xticks()[1:-1])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    ax2.tick_params(axis='x', labelsize = 14)
    
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), shadow=False, ncol=4, fontsize=18, frameon=False)
    

    plt.tight_layout(rect=[0, 0, 1, 1]) 
    
    plt.show()