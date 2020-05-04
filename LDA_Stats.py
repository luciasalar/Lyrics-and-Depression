"""This script examine the relationship between affective quote and depression"""
from quotation_feature import *
import pandas as pd 
import datetime
import collections 
import numpy as np
import scipy.stats
import statsmodels.api as sm
from topic_model import *
from CountVect import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from pprint import pprint
from ruamel import yaml
import datetime
import gc
import nltk
import spacy,en_core_web_sm
from nltk.stem import PorterStemmer

""" This script obtains basic statistics for lyrics and quotes, regression and LDA model
get_noun_trunk_lda, change define which component in the sentence you want to use in LDA"""
def load_experiment(path_to_experiment):
    #load experiment 
    data = yaml.safe_load(open(path_to_experiment))
    return data

class stats:
    def __init__(self, days):

       self.qd = QuotationFeaDynamic()
       self.days = days
       self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'

    def get_data_by_day(self):
        """get posts from the recent X days"""

        sorted_quote = self.qd.get_relative_day(self.days)
        return sorted_quote

    def count_label(self):
        """count number of posts in the past x days"""

        sorted_quote = self.get_data_by_day()
        label_c = sorted_quote.groupby('label').count()
        return label_c

    def quote_cor(self):
        """count average sentiment peruser, correlation between affective quote and depression"""
        sorted_quote = self.get_data_by_day()
        #sorted_quote.to_csv(self.path + 'temp.csv')

        pos1 = sorted_quote.loc[sorted_quote['sentiment_sum'] > 0]
        neg1 = sorted_quote.loc[sorted_quote['sentiment_sum'] < 0]
        neu1 = sorted_quote.loc[sorted_quote['sentiment_sum'] == 0]

        pos = pos1.groupby(['userid']).size().to_frame(name='pos').reset_index()
        neg = neg1.groupby(['userid']).size().to_frame(name='neg').reset_index()
        neu = neu1.groupby(['userid']).size().to_frame(name='neu').reset_index()
        
        # get non-orginal content
        quote = sorted_quote.loc[sorted_quote['label'] != 0]
        # sorted_quote.to_csv(self.path + 'quotation_f.csv')
        #  depression score 
        #cesd = quote.drop_duplicates(subset='userid', keep="first")
        only2011 = pd.read_csv(self.path + 'only2011_users.csv')
        cesd = only2011[['userid', 'cesd_sum']]
        # count number of non-orginal content
        all_count = quote.groupby(['userid']).size().to_frame(name='all_count').reset_index()
        #all_count = cesd.merge(all_count,on='userid')
        # count number of positive non-original content
        positive = quote.loc[quote['sentiment_sum'] > 0]
        per_pos = positive.groupby(['userid']).size().to_frame(name='pos_counts').reset_index()

        negative = quote.loc[quote['sentiment_sum'] < 0]
        per_nega = negative.groupby(['userid']).size().to_frame(name='nega_counts').reset_index()

        neutral = quote.loc[quote['sentiment_sum'] == 0]
        per_neu = neutral.groupby(['userid']).size().to_frame(name='neu_counts').reset_index()

        lyrics = quote.loc[quote['label'] == 2]
        lyrics_c = lyrics.groupby(['userid']).size().to_frame(name='lyrics').reset_index()

        pos_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] > 0]
        neg_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] < 0]
        neu_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] == 0]
        
        pos_lyrics = pos_lyrics1.groupby(['userid']).size().to_frame(name='pos_lyrics').reset_index()
        neg_lyrics = neg_lyrics1.groupby(['userid']).size().to_frame(name='neg_lyrics').reset_index()
        neu_lyrics = neu_lyrics1.groupby(['userid']).size().to_frame(name='neu_lyrics').reset_index()
        
        quote2 = quote.loc[quote['label'] == 1]
        pos_quote = quote2.loc[quote2['sentiment_sum'] > 0]
        neg_quote = quote2.loc[quote2['sentiment_sum'] < 0]
        neu_quote = quote2.loc[quote2['sentiment_sum'] == 0]

        pos_quote = pos_quote.groupby(['userid']).size().to_frame(name='pos_quote').reset_index()
        neg_quote = neg_quote.groupby(['userid']).size().to_frame(name='neg_quote').reset_index()
        neu_quote = neu_quote.groupby(['userid']).size().to_frame(name='neu_quote').reset_index()

        quote3 = quote2.groupby(['userid']).size().to_frame(name='quote').reset_index()

        print('Among {} non-orginal content, positive content {}, negative content {}, neutral content {},  number of quotes: {},  numeber of lyrics: {}, positive lyrics: {}, negative lyrics {}, neutral lyrics {} '.format(quote.shape[0], positive.shape[0], negative.shape[0], neutral.shape[0], quote2.shape[0], lyrics.shape[0], pos_lyrics1.shape[0], neg_lyrics1.shape[0], neu_lyrics1.shape[0]))

        return per_pos, per_nega, per_neu, all_count, lyrics_c, quote3, pos_lyrics, neg_lyrics, neu_lyrics, pos_quote, neg_quote, neu_quote, pos, neg, neu, cesd
        

    def quote_cor_magnitude(self):
        """count average sentiment peruser, correlation between affective quote and depression"""
        sorted_quote = self.get_data_by_day()
        #sorted_quote.to_csv(self.path + 'temp.csv')

        pos1 = sorted_quote.loc[sorted_quote['sentiment_sum'] > 0]
        neg1 = sorted_quote.loc[sorted_quote['sentiment_sum'] < 0]
        neu1 = sorted_quote.loc[sorted_quote['sentiment_sum'] == 0]

        pos = pos1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='pos_mag').reset_index()
        neg = neg1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neg_mag').reset_index()
        neu = neu1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neu_mag').reset_index()

        only2011 = pd.read_csv(self.path + 'only2011_users.csv')
        cesd = only2011[['userid', 'cesd_sum']]
        
        # get non-orginal content
        quote = sorted_quote.loc[sorted_quote['label'] != 0]

        lyrics = quote.loc[quote['label'] == 2]
        lyrics_c = lyrics.groupby(['userid']).size().to_frame(name='lyrics').reset_index()

        pos_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] > 0]
        neg_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] < 0]
        neu_lyrics1 = lyrics.loc[lyrics['sentiment_sum'] == 0]
        
        pos_lyrics = pos_lyrics1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='pos_mag_lyrics').reset_index()
        neg_lyrics = neg_lyrics1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neg_mag_lyrics').reset_index()
        neu_lyrics = neu_lyrics1.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neu_mag_lyrics').reset_index()
        
        quote2 = quote.loc[quote['label'] == 1]
        pos_quote = quote2.loc[quote2['sentiment_sum'] > 0]
        neg_quote = quote2.loc[quote2['sentiment_sum'] < 0]
        neu_quote = quote2.loc[quote2['sentiment_sum'] == 0]

        pos_quote = pos_quote.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='pos_mag_quote').reset_index()
        neg_quote = neg_quote.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neg_mag_quote').reset_index()
        neu_quote = neu_quote.groupby(['userid'])['sentiment_sum'].mean().to_frame(name='neu_mag_quote').reset_index()

        #quote3 = quote2.groupby(['userid']).size().to_frame(name='quote').reset_index()

        #print('Among {} non-orginal content, positive content {}, negative content {}, neutral content {},  number of quotes: {},  numeber of lyrics: {}, positive lyrics: {}, negative lyrics {}, neutral lyrics {} '.format(quote.shape[0], positive.shape[0], negative.shape[0], neutral.shape[0], quote2.shape[0], lyrics.shape[0], pos_lyrics1.shape[0], neg_lyrics1.shape[0], neu_lyrics1.shape[0]))

        return pos_lyrics, neg_lyrics, neu_lyrics, pos_quote, neg_quote, neu_quote, pos, neg, neu, cesd
        
    def get_count_quote(self):
        '''Here we see the count of valenced post in each user'''
        p, nega, neutral, all_count, ly, quo, pos_ly, neg_ly, neu_ly, pos_quo, neg_quo, neu_quo, pos, neg, neu, cesd = self.quote_cor()

         # merge all the counts as feature '''
        quotation_fea = p.merge(nega, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neutral, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(all_count, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_ly, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(cesd, on='userid', how='outer')
        quotation_fea = quotation_fea[['userid', 'pos_counts', 'nega_counts', 'neu_counts','all_count','cesd_sum', 'lyrics','quote', 'pos_lyrics', 'neg_lyrics', 'neu_lyrics', 'pos_quote', 'neg_quote', 'neu_quote', 'pos', 'neg', 'neu']]
        quotation_fea['pos_quo_ratio'] = quotation_fea['pos_quote']/quotation_fea['all_count']
        quotation_fea['neg_quo_ratio'] = quotation_fea['neg_quote']/quotation_fea['all_count']
        quotation_fea['neu_quo_ratio'] = quotation_fea['neu_quote']/quotation_fea['all_count']
        quotation_fea['neg_pos_quo_ratio'] = quotation_fea['neg_quote']/quotation_fea['pos_quote']

        quotation_fea['pos_lyr_ratio'] = quotation_fea['pos_lyrics']/quotation_fea['all_count']
        quotation_fea['neg_lyr_ratio'] = quotation_fea['neg_lyrics']/quotation_fea['all_count']
        quotation_fea['neu_lyr_ratio'] = quotation_fea['neu_lyrics']/quotation_fea['all_count']
        quotation_fea['neg_pos_lyr_ratio'] = quotation_fea['neg_lyrics']/quotation_fea['pos_lyrics']

        #quotation_fea['neg_pos_ratio'] = quotation_fea['nega']/quotation_fea['pos_lyrics']

        quotation_fea = quotation_fea.fillna(0)
        quotation_fea['all_post_count'] = quotation_fea['pos'] + quotation_fea['neg'] + quotation_fea['neu'] 
        quotation_fea = quotation_fea[quotation_fea['all_post_count'] > 0]
        quotation_fea['non_origin_ratio'] = quotation_fea['all_count']/quotation_fea['all_post_count']
        quotation_fea['neg_ratio'] = quotation_fea['neg']/quotation_fea['all_post_count']
        quotation_fea['pos_ratio'] = quotation_fea['pos']/quotation_fea['all_post_count']
        quotation_fea['neu_ratio'] = quotation_fea['neu']/quotation_fea['all_post_count']
        quotation_fea['neg_pos_ratio'] = quotation_fea['neg']/quotation_fea['pos']

        quotation_fea['lyr_ratio'] = quotation_fea['lyrics']/quotation_fea['all_post_count']
        #quotation_fea['lyr_ratio'] = quotation_fea['']/quotation_fea['all_post_count']
        return quotation_fea

    def get_magnitude(self):
        '''Here we see the count of valenced post in each user'''
        pos_lyrics, neg_lyrics, neu_lyrics, pos_quo, neg_quo, neu_quo, pos, neg, neu, cesd = self.quote_cor_magnitude()


        quotation_fea = pos_lyrics.merge(neg_lyrics, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_lyrics, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu_quo, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(pos, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neg, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(cesd, on='userid', how='outer')
        quotation_fea = quotation_fea[['userid', 'pos_mag_lyrics', 'neg_mag_lyrics', 'neu_mag_lyrics', 'pos_mag_quote', 'neg_mag_quote', 'neu_mag_quote', 'pos_mag', 'neg_mag', 'neu_mag', 'cesd_sum']]


        #quotation_fea['neg_pos_ratio'] = quotation_fea['nega']/quotation_fea['pos_lyrics']

        quotation_fea = quotation_fea.fillna(0)
     
        #quotation_fea['lyr_ratio'] = quotation_fea['']/quotation_fea['all_post_count']
        return quotation_fea

    def regression(self, pre_var, data):
        s = stats(self.days) 
        #data = s.get_count_quote()
   
        X = data[pre_var]
        y = data["cesd_sum"]
        model = sm.OLS(y, X).fit()
        print(model.summary())
        # pls2 = PLSRegression(n_components=2)
        # pls2.fit(X, y)
        # Y_pred = pls2.predict(X)
        # # Calculate scores
        # score = r2_score(y, Y_pred)
        # mse = mean_squared_error(y, Y_pred)
        # print('score is {}, R square is {}'.format(score, mse))

        #predictions = model.predict(X)

    def process_text_lda(self, post):
        '''porcess text and tag them with high/low symptoms '''
        nonOrg = post.loc[post['label'] == 1]
        lyrics = post.loc[post['label'] == 2]

        topic = LDATopicModel()
        c = Count_Vect()

        text = lyrics[['text', 'userid']]
        text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
        text = c.get_precocessed_text(text)
        text['text'] = text['text'].apply(lambda x: x.split())

        return text

    def get_bigram_lda(self, post):
        """get bigrams, replace process_text_lda"""
        nonOrg = post.loc[post['label'] == 1]
        lyrics = post.loc[post['label'] == 2]

        topic = LDATopicModel()
        c = Count_Vect()

        text = lyrics[['text', 'userid']]
        text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
        text = c.get_precocessed_text(text)
        text2 = text['text'].tolist()

        new_l = []
        for row in text2:
            bigrm = list(nltk.bigrams(row.split()))
            bi_map = map('_'.join, bigrm)
            process_bi = [', '.join(str(b) for b in bi_map)]
            new_l.append(process_bi)

        text['text'] = new_l

        return text



    def text_tag_high_symptoms(self, text):
        """convert pd to dictionary first then replace the new list, convert it back to pd"""

        #convert df to dictionary
        tagged_dict = {}
        
        for word_l, userid in zip(text['text'], text['post_id']):
            
        #tagging each word in the list
            new_word_l = []
            for w in word_l:
                new_w = w + '_H'
                new_word_l.append(new_w)
            
            tagged_dict[userid] = new_word_l

        text_df = pd.DataFrame(tagged_dict.items(), columns=['post_id', 'text'])
        return text_df

    def text_tag_low_symptoms(self, text):
        """convert pd to dictionary first then replace the new list, convert it back to pd"""

        #convert df to dictionary
        tagged_dict = {}
        
        for word_l, userid in zip(text['text'], text['post_id']):
            
        #tagging each word in the list
            new_word_l = []
            for w in word_l:
                new_w = w + '_L'
                new_word_l.append(new_w)
            
            tagged_dict[userid] = new_word_l

        text_df = pd.DataFrame(tagged_dict.items(), columns=['post_id', 'text'])
        return text_df


    def get_lda_individual_tagged(self, topic_num, alpha, eta):
        """get lda topics for lyrics"""
        topic = LDATopicModel()
        # s = stats(180)
        post = self.get_data_by_day()
        high_symp = post[post['cesd_sum'] > 22]
        low_symp = post[post['cesd_sum'] <= 22]

        text_high = self.process_text_lda(high_symp)
        text_low = self.process_text_lda(low_symp)
                
        text_high['post_id'] = text_high.index
        text_low['post_id'] = text_low.index
        tagged_high = self.text_tag_high_symptoms(text_high)
        tagged_low = self.text_tag_low_symptoms(text_low)
        text = tagged_high.append(tagged_low)

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]
        model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
        return model, coherence

    def text_tag_high_symptoms2(self, text):
        """convert pd to dictionary first then replace the new list, convert it back to pd"""

        #convert df to dictionary
        tagged_dict = {}
        
        for word_l, userid in zip(text['text'], text['post_id']):
            
        #tagging each word in the list
            new_word_l = []
            for w in word_l[0].split(','):
                new_w = w + '_H'
                new_word_l.append(new_w)
            
            tagged_dict[userid] = new_word_l

        text_df = pd.DataFrame(tagged_dict.items(), columns=['post_id', 'text'])
        return text_df

    def text_tag_low_symptoms2(self, text):
        """convert pd to dictionary first then replace the new list, convert it back to pd"""

        #convert df to dictionary
        tagged_dict = {}
        
        for word_l, userid in zip(text['text'], text['post_id']):
            
        #tagging each word in the list
            new_word_l = []
            for w in word_l[0].split(','):
                new_w = w + '_L'
                new_word_l.append(new_w)
            
            tagged_dict[userid] = new_word_l

        text_df = pd.DataFrame(tagged_dict.items(), columns=['post_id', 'text'])
        return text_df


    def get_lda_individual_tagged_bigram(self, topic_num, alpha, eta):
        """get lda topics for lyrics"""
        topic = LDATopicModel()
        # s = stats(180)
        post = self.get_data_by_day()
        high_symp = post[post['cesd_sum'] > 22]
        low_symp = post[post['cesd_sum'] <= 22]

        text_high = self.get_bigram_lda(high_symp)
        text_low = self.get_bigram_lda(low_symp)

        #print(text_high.head)
        text_high['post_id'] = text_high.index
        text_low['post_id'] = text_low.index
        tagged_high = self.text_tag_high_symptoms2(text_high)
        tagged_low = self.text_tag_low_symptoms2(text_low)
        #print(tag)

        text = tagged_high.append(tagged_low)

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]
        model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
        return model, coherence

    def get_noun_trunk_lda(self, post, text):
        """get noun trunks for the lda model, change noun and verb part to decide what you want to use as input for LDA"""
        c = Count_Vect()
        ps = PorterStemmer()
        nonOrg = post.loc[post['label'] == 1]
        lyrics = post.loc[post['label'] == 2]
        #lyrics = nonOrg

        if text in 'lyrics':
            text = lyrics[['text', 'userid', 'textID', 'label']]
            text = text.rename(columns={"text": "text_pre"})
            #print(text['label'])
            text['text'] = text['text_pre'].apply(lambda x: c.remove_noise(str(x.lower())))
            text = c.get_precocessed_text_simple(text)

            #find nound trunks
            nlp = en_core_web_sm.load()
            all_extracted = []
            for post in text['text']:
                doc = nlp(post)
                nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split() 
                #nouns = ' '.join(str('_'.join(str(w) for w in n)) for n in doc.noun_chunks).split() 
                verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split() 
                adj = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'ADJ').split() 
                all_w = nouns + verbs + adj
                all_extracted.append(all_w)
            text['text'] = all_extracted

        elif text in 'quotes':
            text = nonOrg[['text', 'userid', 'textID', 'label']]
            text = text.rename(columns={"text": "text_pre"})
            text['text'] = text['text_pre'].apply(lambda x: c.remove_noise(str(x.lower())))
            text = c.get_precocessed_text_simple(text)

            #find nound trunks
            nlp = en_core_web_sm.load()
            all_extracted = []
            for post in text['text']:
                doc = nlp(post)
                nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split() 
                #nouns = ' '.join(str('_'.join(str(w) for w in n)) for n in doc.noun_chunks).split() 
                verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split() 
                adj = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'ADJ').split() 
                all_w = nouns + verbs + adj
                all_extracted.append(all_w)
            text['text'] = all_extracted


        return text

    def get_lda_individual_tagged_NV(self, topic_num, alpha, eta, text1):
        """get lda topics for lyrics, noun and verbs and adjs"""
        topic = LDATopicModel()
        # s = stats(180)
        post = self.get_data_by_day()
        high_symp = post[post['cesd_sum'] > 22]
        low_symp = post[post['cesd_sum'] <= 22]

        text_high = self.get_noun_trunk_lda(high_symp, text1)
        text_low = self.get_noun_trunk_lda(low_symp, text1)

        #print(text_high.head)
        text_high['post_id'] = text_high['textID']
        text_low['post_id'] = text_low['textID']
        tagged_high = self.text_tag_high_symptoms(text_high)
        tagged_low = self.text_tag_low_symptoms(text_low)
        #print(tag)

        text = tagged_high.append(tagged_low)
        text2 = text_high.append(text_low)

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]
        model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
        return model, coherence, bow_corpus, text2

    def get_lda_tagged_NV_loop(self, text): 
        """big loop for lda topics with noun and verb and high/low group tagged"""
        path = '/disk/data/share/s1690903/QuoteAndDepression/'
        experiment = load_experiment(path + './experiment/experiment.yaml')

        file_exists = os.path.isfile(path + 'result/lda_tagged_results_nva_quote.csv')
        f = open(path + 'result/lda_tagged_results_nva_quote.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['topic_number'] + ['cohenrence_score'] + ['alpha'] + ['eta'] + ['time'] + ['topic_words'])
            
        # topics, model, coherence = s.get_lda(5)
        topic = LDATopicModel()
        # s = stats(180)
        post = self.get_data_by_day()
        high_symp = post[post['cesd_sum'] > 22]
        low_symp = post[post['cesd_sum'] <= 22]

        text_high = self.get_noun_trunk_lda(high_symp, text)
        text_low = self.get_noun_trunk_lda(low_symp, text)

        #print(text_high.head)
        text_high['post_id'] = text_high.index
        text_low['post_id'] = text_low.index
        tagged_high = self.text_tag_high_symptoms(text_high)
        tagged_low = self.text_tag_low_symptoms(text_low)
        #print(tag)

        text = tagged_high.append(tagged_low)

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        
        for topic_num in experiment['lda']['topics']: 
            for alpha in experiment['lda']['alphas']:
                for eta in experiment['lda']['etas']:

                    print('running model..., alphas is {}, eta is {}'.format(alpha, eta))
                    print(text.head(10))
                    
                    model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
                    
                    result_row = [[topic_num, coherence, alpha, eta, str(datetime.datetime.now()), model.print_topics()]]
                    gc.collect()
                    writer_top.writerows(result_row)
        f.close()




    def get_lda_tagged(self): 
        """big loop for lda topics with high/low group tagged """
        path = '/disk/data/share/s1690903/QuoteAndDepression/'
        experiment = load_experiment(path + './experiment/experiment.yaml')

        file_exists = os.path.isfile(path + 'result/lda_tagged_results.csv')
        f = open(path + 'result/lda_tagged_results.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['topic_number'] + ['cohenrence_score'] + ['alpha'] + ['eta'] + ['time'] + ['topic_words'])
            f.close()
        # topics, model, coherence = s.get_lda(5)
        topic = LDATopicModel()
        # s = stats(180)
        post = self.get_data_by_day()
        high_symp = post[post['cesd_sum'] > 22]
        low_symp = post[post['cesd_sum'] <= 22]

        text_high = self.process_text_lda(high_symp)
        text_low = self.process_text_lda(low_symp)
        text_high['post_id'] = text_high.index
        text_low['post_id'] = text_low.index
        tagged_high = self.text_tag_high_symptoms(text_high)
        tagged_low = self.text_tag_low_symptoms(text_low)
        text = tagged_high.append(tagged_low)

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        f = open(path + 'lda_tagged_results.csv', 'a')
        for topic_num in experiment['lda']['topics']: 
            for alpha in experiment['lda']['alphas']:
                for eta in experiment['lda']['etas']:

                    print('running model..., alphas is {}, eta is {}'.format(alpha, eta))
                    print(text.head(10))
                    
                    model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
                    
                    result_row = [[topic_num, coherence, alpha, eta, str(datetime.datetime.now()), model.print_topics()]]
                    gc.collect()
                    writer_top.writerows(result_row)
        f.close()



    def get_lda_individual(self, topic_num, alpha, eta, text):
        """get individual lda model"""

        s = stats(360)
        post = s.get_data_by_day()

        nonOrg = post.loc[post['label'] == 1]
        lyrics = post.loc[post['label'] == 2]

        topic = LDATopicModel()
        c = Count_Vect()
        
        # change here to define lyrics or quote
        if text in 'lyrics':
            text = lyrics[['text', 'userid']]
            text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
            text = c.get_precocessed_text(text)
            text['text'] = text['text'].apply(lambda x: x.split())

            dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
            bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]
            model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)

        elif text in 'quotes':
            text = nonOrg[['text', 'userid']]
            text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
            text = c.get_precocessed_text(text)
            text['text'] = text['text'].apply(lambda x: x.split())

            dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
            bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]
            model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
        
        # model evaluation 
        return model, coherence

    def get_lda(self): 

        path = '/disk/data/share/s1690903/QuoteAndDepression/'
        experiment = load_experiment(path + './experiment/experiment.yaml')

        file_exists = os.path.isfile(path + 'result/lda_results.csv')
        f = open(path + 'result/lda_results.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['topic_number'] + ['cohenrence_score'] + ['alpha'] + ['eta'] + ['time'] + ['topic_words'])
            f.close()

        # topics, model, coherence = s.get_lda(5)
        post = self.get_data_by_day()
        nonOrg = post.loc[post['label'] == 1]
        lyrics = post.loc[post['label'] == 2]

        topic = LDATopicModel()
        c = Count_Vect()

        text = lyrics[['text', 'userid']]
        text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
        text = c.get_precocessed_text(text)
        text['text'] = text['text'].apply(lambda x: x.split())

        dictionary = gensim.corpora.Dictionary(text['text'])# generate dictionary
        bow_corpus = [dictionary.doc2bow(doc) for doc in text['text']]

        f = open(path + 'lda_tagged_results.csv', 'a')
        for topic_num in experiment['lda']['topics']: 
            for alpha in experiment['lda']['alphas']:
                for eta in experiment['lda']['etas']:

                    print('running model..., alphas is {}, eta is {}'.format(alpha, eta))
                    print(text.head(10))
                    
                    model, coherence = topic.get_lda_score_eval(dictionary, bow_corpus, topic_num, alpha, eta)
                    
                    result_row = [[topic_num, coherence, alpha, eta, str(datetime.datetime.now()), model.print_topics()]]
                    gc.collect()
                    writer_top.writerows(result_row)
        f.close()

    def get_dominance_topic(self, topic_num, alpha, eta, text):
        model, coherence, corpus, text = s.get_lda_individual_tagged_NV(topic_num, alpha, eta, text)
        t = LDATopicModel()
        df_topic_sents_keywords = t.format_topics_sentences(model, corpus)

        df_dominant_topic = df_topic_sents_keywords.reset_index()
        text = text.reset_index()
        sent_topics_df = pd.concat([df_dominant_topic, text], axis=1)
        sent_topics_df = sent_topics_df[['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords','text_pre', 'text', 'userid','textID', 'label']]

        path = '/disk/data/share/s1690903/QuoteAndDepression/result/'
        sent_topics_df.to_csv(path + 'topic_dominance_{}.csv'.text)

        return sent_topics_df


#topic analysis
s = stats(360)

# get result for quote
s.get_lda_tagged_NV_loop('quote')# select lyrics or quotes
#select best model and parameters
#sent_topics_df = s.get_dominance_topic(15, 0.05, 0.9, 'lyrics')
sent_topics_df = s.get_dominance_topic(15, 0.07, 0.9, 'quotes')
sent_topics_df.groupby('Dominant_Topic').count()



# s = stats(360) 
# fea = s.get_count_quote()
# fea2 = s.get_magnitude()


# fea = fea.drop(['cesd_sum'], axis=1)
# all_fea = fea.merge(fea2, on = 'userid', how = 'inner')

# all_fea_sel = all_fea[['neg_quo_ratio', 'neu_quo_ratio', 'pos_lyr_ratio', 
# 'pos_ratio','neu_ratio','pos_mag_lyrics', 'neg_mag_lyrics', 'neu_mag_lyrics', 'pos_mag_quote', 'neg_mag_quote', 'neu_mag_quote', 'pos_mag', 'neg_mag', 'neu_mag']]

# #cor = all_fea_sel.corr()

# predictive_var = ['all_post_count', 'pos_quo_ratio', 'neg_quo_ratio', 'neu_quo_ratio', 'pos_lyr_ratio', 'neg_lyr_ratio','neu_lyr_ratio', 'non_origin_ratio', 'neg_ratio', 
# 'pos_ratio','neu_ratio','pos_mag_lyrics', 'neg_mag_lyrics', 'neu_mag_lyrics', 'pos_mag_quote', 'neg_mag_quote', 'neu_mag_quote', 'pos_mag', 'neg_mag', 'neu_mag']

# s.regression(predictive_var, all_fea)


#the correlation between number of posting frequency and cesd are correlated in differe time scale
if __name__ == "__main__":

    s = stats(360) 
    fea = s.get_count_quote()

    scipy.stats.pearsonr(fea['nega_counts'], fea['cesd_sum']) 
    # (0.10215625183027725, 0.005473527850170207)
    scipy.stats.pearsonr(fea['all_count'], fea['cesd_sum'])  
    #(0.11208478507195666, 0.0022932566960116025)
    scipy.stats.pearsonr(fea['pos_counts'], fea['cesd_sum']) 
    #(0.11319384743569788, 0.002071866146220976)§
    scipy.stats.pearsonr(fea['lyrics'], fea['cesd_sum']) 
    # (0.09893687229160814, 0.007149905789159595)
    scipy.stats.pearsonr(fea['quote'], fea['cesd_sum']) 
     # (0.09519970308834393, 0.009661757408637353)

    scipy.stats.pearsonr(fea['neg_lyrics'], fea['cesd_sum']) 
    scipy.stats.pearsonr(fea['pos_lyrics'], fea['cesd_sum']) 
    scipy.stats.pearsonr(fea['neu_lyrics'], fea['cesd_sum'])

    scipy.stats.pearsonr(fea['pos_quote_ratio'], fea['cesd_sum'])

    	#let's do a regression 
    #   s = stats(180) 
        # predictive_var = ['pos_counts', 'nega_counts', 'neu_counts','all_count', 'lyrics','quote', 'pos_lyrics', 'neg_lyrics', 'neu_lyrics', 'pos_quote', 'neg_quote', 'neu_quote', 'pos', 'neg', 'neu'] 
        # 
        # # pre_var = ['lyrics', 'pos']
    predictive_var = ['neu', 'lyrics', 'quote'] 
    s.regression(predictive_var, fea)
    	
    # topic model 

    s = stats(360)
    #s.get_lda()
    s.get_lda_individual(10, 0.05, 0.9)

    #run LDA topics 
    s = stats(360)
    s.get_lda_tagged_NV_loop('lyrics')# select lyrics or quotes

    model, coherence, corpus, text = s.get_lda_individual_tagged_NV(15, 0.05, 0.9, 'lyrics')
    t = LDATopicModel()
  
    #top 5 topics 0, 3, 5, 7, 12












