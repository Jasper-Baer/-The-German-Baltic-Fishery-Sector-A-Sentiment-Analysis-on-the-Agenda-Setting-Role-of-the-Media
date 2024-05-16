# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:06:42 2021

@author: Jasper Bär
"""

import pandas as pd 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer  
from nltk.tokenize import sent_tokenize, word_tokenize 
import spacy 

def tokenizer_articles(data, language):
    """
    Tokenizes articles into sentences.
    """
    rows = []
    for _, row in data.iterrows():
        article = row['text']
        sentences = sent_tokenize(article, language=language)
        metadata = row.drop('text')
        metadata['text'] = sentences
        rows.append(metadata)

    sentence_split = pd.DataFrame(rows).reset_index(drop=True)
    return sentence_split


def tokenize_sentences(sentence, language):
    """
    Tokenizes sentences into words.
    """
    sent_tokens = word_tokenize(sentence, language=language)
    return [sent_tokens]
    
def lemmatize_sentences(sentences, language ,remove_stopwords = 'all'):
    """
    Lemmatizes sentences and optionally removes stopwords.
    """
    # initialize list with all negation words which are kept for the negation feature
    neg_words = ["nicht", "nichts", "kein", "keinen", "keine", "keiner", 
                 "keines", "keinem", "keins", "niemals", "nie"]
    
    # initialize list with special charakter to remove
    delims = ["-", "_", "#", "+", "*", "~", "$", "%", "`", "´", "=", "§", 
              "{", "}", "/", "[", "]", "^", "°", str("("), str(")"), str("'"),
              "&", "in+die","zu+die", "@card@",".", ",", ";",":", str("„"), 
              "@ord@",str('"'), "an+die", "von+die", 'bei+die',".", ",", "?",
              "!", ":", ";", 'für+die', '“', '„', '«', '»', 'bu', 'BU','das', 
              'für', 'fuer', 'oz'] 
    
    stopset = stopwords.words(language)
    
    if remove_stopwords in ('all', 'delim'):
        # load set with stopwords and add delimiters to them
        stopset += delims 
    
    elif remove_stopwords in ('all', 'negations'):
        # add negation words from stopword list
        stopset += neg_words
    
    # add token for links and ip to stopword list
    stopset += ['replaced-dns', 'replaced-ip']

    stopset = set(stopset)

    # Load Spacy's German model
    
    if language == 'german':
    
        nlp = spacy.load('de_core_news_sm')
        
    elif language == 'english':
        
        nlp = spacy.load('en_core_news_sm')

    # Lemmatize sentences
    sent_lemmas = [[token.lemma_.lower() for token in nlp(sent)] for sent in sentences]

    if remove_stopwords in ('all', 'delim', 'negations'):
        # remove stopwords from lemmas and (POS)
        sent_lemmas  = [[word for word in sent_lemmas[i] if word not in stopset] for i in range(0, len(sent_lemmas))]
        lemma_str = [" ".join(j) for j in sent_lemmas]
        lemma_str = [sent.replace('„','') for sent in lemma_str]
        lemma_str = [sent.replace('.','') for sent in lemma_str]
        lemma_str = [sent.replace('“','') for sent in lemma_str]
        
    else:
        lemma_str = [" ".join(j) for j in sent_lemmas]
    
    return lemma_str

def average_sentiment(data):
    """
    Calculates the average sentiment from given labels.
    """
    
    if len(data) > 0:
        
        # Count sentences for each label
        pos_sent = len(data[data['Label'] == 2])
        neu_sent = len(data[data['Label'] == 1])
        neg_sent = len(data[data['Label'] == 0])
        
        # Calcualte sentiment index by dividing the difference between the
        # number of postive and negative sentences with the number of all
        # sentneces
        sentiment = (pos_sent - neg_sent)/len(data)
        
        # Calculate the share of positibe, neutral and negative sentences with
        # respect to all sentences
        pos_sent_ratio = pos_sent/len(data)
        neu_sent_ratio = neu_sent/len(data)
        neg_sent_ratio = neg_sent/len(data)
        
        results = [sentiment, [pos_sent, neu_sent, neg_sent], [pos_sent_ratio, neu_sent_ratio, neg_sent_ratio]]
        
    else:
        
        results = None
    
    return(results)    

def transform_tf_idf(data):
    """
    Transforms the data using TF-IDF vectorization.
    """
    
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(data)
    words = vectorizer.get_feature_names()
    dense_word_vec = vecs.todense()
    word_vec_list = dense_word_vec.tolist() 
    word_vec_list = map(sum, zip(* word_vec_list))
    tf_idf_scores = pd.Series(word_vec_list, index = words)
    tf_idf_scores = tf_idf_scores.sort_values(ascending = False)
    
    return(tf_idf_scores)

def fishery_year(data, quota_dates):
    """  
    Reranges the data based on given dates
    
    :data: A Dataframe with articles
    :quota_dates: A list of quota dates that mark the start and end day for
    each fishery year
    """
    
    fishery_years = []
    # years = list(set(data['year']))
    # years.sort()
    
    for idx, date in enumerate(quota_dates[:-1]):
      
        # if year != 2021:
            
        #   start_date = quota_dates[quota_dates.dt.year == year].iloc[0]
        #   end_date = quota_dates[quota_dates.dt.year == year+1].iloc[0]
      
        #   fishery_years.append(data[(data["Date"] >= start_date) & (data["Date"] <= end_date)])
          
      #  else:
            
          start_date = quota_dates[idx]
          end_date = quota_dates[idx+1]
          
          fishery_years.append(data[(data["Date"] >= start_date) & (data["Date"] < end_date)])
      
    return(fishery_years)


def sentiment_index_yearly(data): 
    """
    Calculates the sentiment index for a list of dataframes with articles on a
    yearly basis.
    """
   
    # Initialize dataframes to store results 
    sentiment_data = pd.DataFrame() 
     
    years = list(set(pd.concat(data)['year'])) 
    years.sort() 
     
    # Remove empty dataframes 
    data = [df for df in data if not df.empty] 
     
    for idx, value in enumerate(data): 
      
     # Calculate sentiment index for specified year 
     yearly_data = data[idx] 
     yearly_sentiment  = average_sentiment(yearly_data)  
      
     if yearly_sentiment != None: 
      
         # Add results from sentiment index caculation 
         sentiment = pd.DataFrame({'year':years[idx], 'Sentiment Index': [yearly_sentiment[0]],  
                                   'pos': [yearly_sentiment[1][0]], 'neu': [yearly_sentiment[1][1]], 
                                   'neg': [yearly_sentiment[1][2]], 'pos_sent_ratio': [yearly_sentiment[2][0]], 
                                   'neu_sent_ratio': [yearly_sentiment[2][1]], 'neg_sent_ratio': [yearly_sentiment[2][2]]}) 
     else: 
          
         sentiment = None 
      
     sentiment_data = pd.concat([sentiment_data, sentiment]) 
         
    return(sentiment_data) 
 
def sentiment_index_quarterly(data): 
    """   
    Calculates the sentiment index for a list of dataframes 
    with articles on a quarterly basis.
    
    """ 
    data['Date'] = pd.to_datetime(data['Date'])

    data['year'] = data['Date'].dt.year
    data['quarter'] =data['Date'].dt.quarter
    
    # Create a dictionary to store each year and quarter's data
    year_quarter_dict = {}
    for year in data['year'].unique():
        for quarter in data['quarter'].unique():
            year_quarter_dict[(year, quarter)] = data[(data['year'] == year) & (data['quarter'] == quarter)]
    
   
    # Initialize dataframes to store results 
    sentiment_data = pd.DataFrame() 
     
    def quarter_to_month(quarter):
        return (quarter-1)*3 + 1

    for key, value in year_quarter_dict.items():
        # Calculate sentiment index for specified year and quarter
        quarterly_data = value
        quarterly_sentiment = average_sentiment(quarterly_data)
    
        if quarterly_sentiment is not None:
            # Add results from sentiment index caculation
            year = key[0]
            quarter = key[1]
            month = quarter_to_month(quarter)
    
            sentiment = pd.DataFrame({
                'year': [year],
                'quarter': [quarter],
                'Date': [pd.Timestamp(year=year, month=month, day=1)],  # create a timestamp for the start of the quarter
                'Sentiment Index': [quarterly_sentiment[0]],
                'pos': [quarterly_sentiment[1][0]], 
                'neu': [quarterly_sentiment[1][1]], 
                'neg': [quarterly_sentiment[1][2]], 
                'pos_sent_ratio': [quarterly_sentiment[2][0]], 
                'neu_sent_ratio': [quarterly_sentiment[2][1]], 
                'neg_sent_ratio': [quarterly_sentiment[2][2]]
            })
        else:
            sentiment = None
    
        sentiment_data = pd.concat([sentiment_data, sentiment])
        
    sentiment_data = sentiment_data.sort_values(by='Date').reset_index(drop=True)
         
    return(sentiment_data)    
 
def select_stakeholders(data, stakeholders): 
   """   
   This function selects all texts which include given stakeholders 
    
   :data: A Dataframe which includes a columns with 'lemmas'  
   :stakeholders: A list of strings  
   """ 
     
   idxs = [idx for idx, text in enumerate(data['lemmas']) if any(stakeholder in text for stakeholder in stakeholders)] 
   stakeholder_data = data.iloc[idxs ] 
    
   return(stakeholder_data)