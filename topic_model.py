import warnings
warnings.filterwarnings('always')
import numpy as np
import pandas as pd
import re
import math
#import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from pprint import pprint
import gensim
import pickle
import collections
#import psycopg2
import time
import spacy
from CountVect import *
import tracemalloc
import datetime
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import CoherenceModel


tracemalloc.start()

"""
here we run lda on the extracted entities
"""

class LDATopicModel:
    def __init__(self):
        '''define the main path'''
        #self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'


    def get_score_dict(self, bow_corpus, lda_model_object):
        """
        get lda score for each document
        """
        all_lda_score = {}
        for i in range(len(bow_corpus)):
            lda_score ={}
            for index, score in sorted(lda_model_object[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                lda_score[index] = score
                od = collections.OrderedDict(sorted(lda_score.items()))
            all_lda_score[i] = od
        return all_lda_score


    def pickle_object(self, object, path_to_save):
        """
        pickle object
        """
        with open(self.path, 'wb') as handle:
            pickle.dump(object, handle, protocol = pickle.HIGHEST_PROTOCOL)


    def get_lda_score(self, text, topics_numbers):


        #text = get_liwc_text(365)
        c= Count_Vect() #initialize text preprocessing class
        
        #text['text'] = text['text'].apply(lambda x: c.remove_single_letter(x))
        text['text'] = text['text'].apply(lambda x: x.split())
    
        dictionary = gensim.corpora.Dictionary(text['text']) #generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        print('running LDA...') 
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=10,  update_every=1, random_state = 300)
        #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=2, workers=10, random_state = 300)

        #getting LDA score 
        lda_score_all = self.get_score_dict(bow_corpus, lda_model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)
        all_lda_score_dfT['userid'] = text['userid']

        pprint(lda_model.print_topics())
        all_lda_score_dfT.to_csv(self.path + 'ldaScores{}.csv'.format(str(datetime.datetime.now())))

        return all_lda_score_dfT, lda_model

    def get_lda_score_eval(self, dictionary, bow_corpus, topics_numbers, alpha, eta):

        '''loop through lda parameters and store results'''

        #text = get_liwc_text(365)
       # c= Count_Vect() #initialize text preprocessing class
        #
        #text['text'] = text['text'].apply(lambda x: c.remove_single_letter(x))
        #text['text'] = text['text'].apply(lambda x: x.split())
    
        #dictionary = gensim.corpora.Dictionary(text['text']) #generate dictionary
        #bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        print('running LDA...') 
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=alpha, eta=eta)
        #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=2, workers=10, random_state = 300)

        #getting LDA score 
        # lda_score_all = self.get_score_dict(bow_corpus, lda_model)

        # all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        # all_lda_score_dfT = all_lda_score_df.T
        # all_lda_score_dfT = all_lda_score_dfT.fillna(0)
        # all_lda_score_dfT['userid'] = text['userid']

        pprint(lda_model.print_topics())
       # all_lda_score_dfT.to_csv(self.path + 'ldaScores{}.csv'.format(str(datetime.datetime.now())))

        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        coherence = cm.get_coherence()  # get coherence value
        print('coherence score is {}'.format(coherence))

        

        return lda_model, coherence



    def get_lda_score2(self, text, topics_numbers):
        """bigram LDA"""

        #text = get_liwc_text(365)
        c= Count_Vect() #initialize text preprocessing class
        
        text['text'] = text['text'].apply(lambda x: c.remove_single_letter(x))
        text['text'] = text['text'].apply(lambda x: x.split())
    
        text1 = text['text'].tolist()

        lemmatizer = WordNetLemmatizer()
        text1 = [[lemmatizer.lemmatize(token) for token in doc] for doc in text1]

        bigram = Phrases(text1 , min_count=2)
        for idx in range(len(text1)):
            for token in bigram[text1[idx]]:
                if '_' in token:
                # Token is a bigram, add to document.
                    text1[idx].append(token)
                    print(text1)


        #dictionary = gensim.corpora.Dictionary(text['text']) #generate dictionary
        dictionary = gensim.corpora.Dictionary(text1)
        bow_corpus = [dictionary.doc2bow(doc) for doc in text1]

        print('running LDA...') 
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=10,  update_every=1, random_state = 300)
        #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics= topics_numbers, id2word=dictionary, passes=2, workers=10, random_state = 300)

        #getting LDA score 
        lda_score_all = self.get_score_dict(bow_corpus, lda_model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)
        all_lda_score_dfT['userid'] = text['userid']

        pprint(lda_model.print_topics())
        all_lda_score_dfT.to_csv(self.path + 'ldaScores{}.csv'.format(str(datetime.datetime.now())))

        return all_lda_score_dfT, lda_model

    def format_topics_sentences(self, ldamodel, corpus):
        # Init output, get dominant topic for each document 
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    

        #Add original text to the end of the output
        #contents = pd.Series(data['text'])     
        #contents = contents.reset_index()
        #sent_topics_df = sent_topics_df.reset_index()
        
        return sent_topics_df

    def topics_per_document(self, model, corpus):
        corpus_sel = corpus[0:-1]
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs, wordid_topics, wordid_phivalues = model[corp]
            dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        return(dominant_topics, topic_percentages)


# if __name__ == "__main__":

#     c= Count_Vect()
#     path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
#     text = pd.read_csv(path + 'status_sentiment.csv') 

#     text = text.head(10000)
#     text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
#     text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
#     text = c.get_precocessed_text(text)

#     l = LDATopicModel()
#     topics, model = l.get_lda_score(text, 30)
  

