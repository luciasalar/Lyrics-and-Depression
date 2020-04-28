#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import sklearn
import nltk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer, MissingIndicator
from ruamel import yaml
import datetime
import matplotlib.pyplot as plt
import os
from CountVect import *
from topic_model import *
import time 
import logging
import csv
from sklearn.decomposition import TruncatedSVD
from construct_mood_feature import *
from construct_mood_transition_feature import *
import gc
import datetime
from basic_quo_fea import *
from sklearn.dummy import DummyClassifier
from quotation_feature import *

#to do 

# add mood changes every 7 days, 3 days, mood category as features, mood transitions


def load_experiment(path_to_experiment):
	#load experiment 
	data = yaml.safe_load(open(path_to_experiment))
	return data

class PrepareData():

	def __init__(self, timewindow, step):
		'''define the main path'''
		self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
		
		self.timeRange = 365
		self.participants = pd.read_csv(self.path + 'participants_matched.csv') 
		self.timewindow = timewindow
		self.step = step
		self.mood =  MoodFeature(path = self.path, participants = self.participants)
		self.quote =  QuoteFeatures()
		self.qdyna = QuotationFeaDynamic()

	def liwc_preprocess(self, timeRange):
		'''aggregate text then process text with liwc'''
		#get posts within time range
		#participants = pd.read_csv(self.path + 'participants_matched.csv') 
		
		sentiment_pre = pd.read_csv(self.path + 'status_sentiment.csv')  
		text = mood.get_relative_day(sentiment_pre, timeRange)
		#merge posts with matched participants
		text_merge = pd.merge(self.participants, text, on = 'userid') 
		text_fea = text_merge[['userid','text']]
		#aggregate text    
		'''DONT USE JOIN STRING!! USE STR.CAT OTHERWISE YOU WILL LOSE A LOT OF INFORMATION  '''
		text_fea3 = text_fea.groupby(['userid'])['text'].apply(lambda x: x.str.cat(sep=',')).reset_index()
		text_fea3 = text_fea3.drop_duplicates() #remove duplication
		text_fea3.to_csv(self.path + 'aggregrate_text_{}.csv'.format(timeRange))

		return text_fea3


	def sentiment_data(self):
		'''generate sentiment features in time window X'''
		#read sentiment data
		# mood = MoodFeature(path = self.path, participants = self.participants)
		sentiment_pre = pd.read_csv(self.path + 'status_sentiment.csv')
		sentiment = self.mood.get_relative_day(sentiment_pre, 365)

		sentiment = sentiment[['userid','time','positive','negative']]
		#compute sentiment sum on each post
		sentiment['sentiment_sum'] = sentiment['positive'] + sentiment['negative']

		#compute average sentiment
		s_mean = sentiment.groupby('userid')['sentiment_sum'].mean().to_frame().reset_index() 
		#compute sentiment sd
		s_sd = sentiment.groupby('userid')['sentiment_sum'].std().to_frame().reset_index() 
		#compute number of post
		p_count = sentiment.groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 
		p_count = p_count.rename(columns={"sentiment_sum": "post_c"})
		#count positive and negative sentiment post
		p_neg_count = sentiment.loc[sentiment['sentiment_sum'] < 0].groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 
		p_pos_count = sentiment.loc[sentiment['sentiment_sum'] >= 0].groupby('userid')['sentiment_sum'].count().to_frame().reset_index() 

		p_total = pd.merge(p_count, p_pos_count, on = 'userid', how = 'left') 
		p_total = pd.merge(p_total, p_neg_count, on = 'userid', how = 'left')
		p_total = p_total.rename(columns={"sentiment_sum_x": "pos_count", "sentiment_sum_y": "neg_count"})
		p_total = p_total.fillna(0) #convert NA to 0
		#compute positive and negative ratio
		p_total['pos_per'] = p_total['pos_count']/p_total['post_c']
		p_total['neg_per'] = p_total['neg_count']/p_total['post_c']

		per_fea = p_total[['userid','post_c','pos_per','neg_per']]
		#return df with sentiment features
		mean_sd = pd.merge(s_sd, s_mean, on = 'userid', how = 'left')
		mean_sd = mean_sd.rename(columns={"sentiment_sum_x": "sent_sd", "sentiment_sum_y": "sent_mean"})

		sentiment_fea = pd.merge(mean_sd, per_fea, on = 'userid', how = 'left')
		#rename columns
		sentiment_fea.columns = [str(col) + '_sentiment' for col in sentiment_fea.columns]
		sentiment_fea  = sentiment_fea.rename(columns = {"userid_sentiment":"userid"})
		return sentiment_fea


	# def posting_frequency():
	# 	'''compute posting frequency in the past x days'''
	# 	pass
	def demographic(self):
		'''create demographic features '''
		demographics = pd.read_csv(self.path + 'participants_matched.csv')

	def topic_modeling(self, text, topic_number):
		'''return topic features'''
		t = LDATopicModel() 
		lda_score_df, lda_model = t.get_lda_score(text, topic_number) 
		return lda_score_df, lda_model



	def merge_data(self):

	    '''merging features, LIWC, mood vectors'''
	    c = Count_Vect()
	   
	    #select frequent users, here you need to change the matched user files if selection criteria changed
	    participants = pd.read_csv(self.path + 'participants_matched.csv')
	    participants  = participants[['userid','cesd_sum']]
	    #merge with text feature
	    text = self.liwc_preprocess(self.timeRange)

	    text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
	    text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
	    #text = c.parallelize_dataframe(text, c.get_precocessed_text)
	    text = c.get_precocessed_text(text)

	    #get mood as feature: mood in the past x days
	    # mood_feature, windowSzie = self.mood.get_mood_continous_in_timewindow(365, self.timewindow, self.step)
	    # mood_feature.columns = [str(col) + '_mood' for col in mood_feature.columns]
	    # mood_feature['userid'] = mood_feature.index
	    # all_features = pd.merge(text, mood_feature, on = 'userid')

	    #print(mood_feature['userid']) 
	    
	    ##get mood change: in the past x day, the momentum of mood 
	    mood_change, windowSzie = self.mood.get_mood_change_in_timewindow(365, self.timewindow, self.step)
	    mood_change.columns = [str(col) + '_moodChange' for col in mood_change.columns]
	    mood_change['userid'] = mood_change.index
	    #mood_change = mood_change.fillna(mood_change.mean())
	    all_features = pd.merge(text, mood_change, on = 'userid')

	    ##get quotation vecotr 
	    quote_vector_feature, windowSize = self.qdyna.get_quote_continous_in_timewindow(365, 14, 3)
	    quote_vector_feature.columns = [str(col) + '_quote_vec' for col in quote_vector_feature.columns]
	    quote_vector_feature['userid'] = quote_vector_feature.index
	    # quote_vector_feature = quote_vector_feature.rename(columns = {"_quote_vec":"userid"})

	    # get quote changes
	    quote_change_feature, windowSize = self.qdyna.get_quote_change_in_timewindow(365, 14, 3)
	    quote_change_feature.columns = [str(col) + '_quote_change' for col in quote_change_feature.columns]
	    quote_change_feature['userid'] = quote_change_feature.index
	    quote_features = pd.merge(quote_vector_feature, quote_change_feature, on = 'userid')

	    #load  liwc (including WC)  mood = MoodFeature(path = path, participants = self.participants)
	    liwc = pd.read_csv(self.path + 'liwc_scores.csv')
	    liwc['userid'] = liwc['userid'].apply(lambda x: x.split('.')[0])
	    liwc.columns = [str(col) + '_liwc' for col in liwc.columns]
	    liwc = liwc.rename(columns = {"userid_liwc":"userid"})

	    #load sentiment feature (including post count)
	    sentiment = self.sentiment_data()
	   
	    #topic ratio LDA
	    topic_text = all_features[['userid','text']]
	    topics, ldamodel = self.topic_modeling(topic_text, 30) #topic number 
	    topics.columns = [str(col) + '_topic' for col in topics.columns]
	    topics = topics.rename(columns = {"userid_topic":"userid"})

	    #quotation
	    # q_fea = self.quote.get_count_quote()
	    # q_fea.columns = [str(col) + '_quote' for col in q_fea.columns]
	    # q_fea = q_fea.rename(columns = {"userid_quote":"userid"})

	    #merge all features
	    all_features = pd.merge(liwc, all_features, on = 'userid')
	    all_features2 = pd.merge(all_features, sentiment, on = 'userid')
	    all_features2 = pd.merge(all_features2, topics, on = 'userid')
	    # all_features2 = pd.merge(all_features2, q_fea, on = 'userid')
	    all_features2 = pd.merge(all_features2, quote_features, on = 'userid')
	    feature_cesd = pd.merge(all_features2, participants, on = 'userid')

	    return feature_cesd

	

	def get_y(self, feature_df):
		'''get y '''
		y = feature_df['cesd_sum']
		return y

	def recode_y(self, y, threshold):
		'''recode y to binary according to a threshold'''
		new_labels = []
		for i in y:
			if i <= threshold:
				i = 0
			if i > threshold:
				i = 1
			new_labels.append(i)
		return new_labels

	def pre_train(self):
		'''merge data, get X, y and recode y '''
		f = self.merge_data()
		y_cesd = self.get_y(f)
		y_recode = self.recode_y(y_cesd, 22) 
		X = f.drop(columns=['userid', 'cesd_sum'])
		return X, y_recode

	def get_train_test_split(self):
		'''split 10% holdout set, then split train test with the rest 90%, stratify splitting'''
		X, y = self.pre_train()
			# get 10% holdout set for testing
		X_train1, X_final_test, y_train1, y_final_test = train_test_split(X, y, test_size=0.10, random_state = 2020, stratify = y)

			#split train test in the rest of 90% 
		X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.30, random_state = 2020, stratify = y_train1)
		print(X_train.shape, X_test.shape)
		return X_train, X_test, y_train, y_test, y_final_test, X_final_test



class ColumnSelector(BaseEstimator, TransformerMixin):
	'''feature selector for pipline (pandas df format) '''
	def __init__(self, columns):
		self.columns = columns

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		assert isinstance(X, pd.DataFrame)

		try:
		    return X[self.columns]
		except KeyError:
		    cols_error = list(set(self.columns) - set(X.columns))
		    raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

	def get_feature_names(self):
		return self.columns.tolist
		    


class TrainingClassifiers: 
	
	def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list, tfidf_words):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test= y_test
		self.parameters = parameters
		self.features_list = features_list
		self.tfidf_words = tfidf_words 

	def select_features(self):
		'''
		select columns with names in feature list then convert it to a transformer object 
		features_list is in the dictionary 
		'''
		fea_list = []
		for fea in self.features_list: #select column names with keywords in dict
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)
		#flatten a list
		flat = [x for sublist in fea_list for x in sublist]
		#convert to transformer object
		#selected_features = FunctionTransformer(lambda x: x[flat], validate=False)

		return flat

	def get_other_feature_names(self):
		fea_list = []
		for fea in self.features_list: #select column names with keywords in dict
			f_list = [i for i in self.X_train.columns if fea in i]
			fea_list.append(f_list)
		#flatten a list
		flat = [x for sublist in fea_list for x in sublist]
		#convert to transformer object
		return flat


	def setup_pipeline(self, classifier):
		'''set up pipeline'''
		features_col = self.get_other_feature_names()


		pipeline = Pipeline([
			#ColumnSelector(columns = features_list),
		    
		    ('feats', FeatureUnion([
		  #   #generate count vect features
		        ('text', Pipeline([

		            ('selector', ColumnSelector(columns='text')),
		            #('cv', CountVectorizer()),
		            ('tfidf', TfidfVectorizer(max_features = self.tfidf_words, ngram_range = (1,3), stop_words ='english', max_df = 0.50, min_df = 0.0025)),
		             ])),
		  # # select other features, feature sets are defines in the yaml file
		 

		 		('other_features', Pipeline([

		 			('selector',  ColumnSelector(columns = features_col)),
		 			('impute', SimpleImputer(strategy='mean')), #impute nan with mean
		 		])),

		     ])),


		       ('clf', Pipeline([  
		       # ('impute', SimpleImputer(strategy='mean')), #impute nan with mean 
		       ('scale', StandardScaler(with_mean=False)),  #scale features
		        ('classifier',  classifier),  #classifier
		   
		         ])),
		])
		return pipeline

	

	def training_models(self, pipeline):
		'''train models with grid search'''
		grid_search_item = GridSearchCV(pipeline, self.parameters, cv = 5, scoring='accuracy')
		grid_search = grid_search_item.fit(self.X_train, self.y_train)
		
		return grid_search

	def test_model(self, path, classifier):
		'''test model and save data'''
		start = time.time()
		#training model
		print('getting pipeline...')
		#the dictionary returns a list, here we extract the string from list use [0]
		pipeline = self.setup_pipeline(eval(classifier)())

		print('features', self.features_list)
		grid_search = self.training_models(pipeline)
		#make prediction
		print('prediction...')
		y_true, y_pred = self.y_test, grid_search.predict(self.X_test)
		report = classification_report(y_true,y_pred, output_dict=True)
		#store prediction result
		y_pred_series = pd.DataFrame(y_pred)
		result = pd.concat([pd.Series(y_true).reset_index(drop=True), y_pred_series], axis = 1)
		result.columns = ['y_true', 'y_pred']
		#result.to_csv(path + 'results/best_result2.csv' )
		end = time.time()
		#print('running time:{}, fscore:{}'.format(end-start, fscore))

		return report,grid_search,pipeline

def get_liwc_text(timeRange):  
    '''run this to get text for liwc, you need to define the time range in days '''
    prepare = PrepareData()
    sen = prepare.sentiment_data()
    liwc_text = prepare.liwc_preprocess(timeRange)
    liwc_text.to_csv(prepare.path + 'liwc_text.csv')
    return liwc_text

def get_separate_text_file(timeRange):
	''' prepare separate text file for liwc'''
	liwc = get_liwc_text(timeRange)
	prepare = PrepareData()
	for index, row in liwc.iterrows():
		out_file = open(prepare.path + 'liwc_text/{}.txt'.format(row['userid']), 'a')
		out_file.write(row['text'])
		out_file.close()

# liwc_score = pd.read_csv(prepare.path + 'liwc_scores.csv')
# liwc_score['userid'] = liwc_score['userid'].apply(lambda x: x.split('.')[0])


def loop_the_grid(MoodslideWindow):
	'''loop parameters in the environment file '''

	path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
	experiment = load_experiment(path + '../experiment/experiment_quotation.yaml')

	file_exists = os.path.isfile(path + 'results/quotation_all.csv')
	f = open(path + 'results/quotation_all.csv' , 'a')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	if not file_exists:
		writer_top.writerow(['best_scores'] + ['best_parameters'] +['report'] +['time'] + ['model'] +['feature_set'] +['tfidf_words'] + ['timewindow'] + ['step'] + ['MoodslideWindow'])
		f.close()

	for classifier in experiment['experiment']:
		for timewindow in experiment['timewindow']:
			for step in experiment['step']:
				if step < timewindow:
					# prepare environment 
					prepare = PrepareData(timewindow = timewindow, step = step)
					
					# split data
					X_train, X_test, y_train, y_test, y_final_test, X_final_test = prepare.get_train_test_split()
					X_train.to_csv(path + 'train_feature.csv')
					for feature_set, features_list in experiment['features'].items(): #loop feature sets
						for tfidf_words in experiment['tfidf_features']['max_fea']: #loop tfidf features
							

							f = open(prepare.path + 'results/quotation_all.csv' , 'a')
							writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)

							parameters = experiment['experiment'][classifier]
							print('parameters are:', parameters)
							training = TrainingClassifiers(X_train = X_train, y_train=y_train, X_test=X_test, y_test =y_test, parameters =parameters, features_list =features_list, tfidf_words=tfidf_words)
							
							report, grid_search, pipeline = training.test_model(prepare.path, classifier)

							result_row = [[grid_search.best_score_, grid_search.best_params_, pd.DataFrame(report), str(datetime.datetime.now()), classifier, features_list, tfidf_words,timewindow, step, MoodslideWindow]]

							writer_top.writerows(result_row)

							f.close()
							gc.collect()
					
loop_the_grid('30_days')

# dummy classifier
# def dummy_classifier(MoodslideWindow):
'''loop parameters in the environment file '''

# print('quotation ************')
# path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
# experiment = load_experiment(path + '../experiment/experiment_quotation.yaml')

# timewindow = 14
# step = 3
# # prepare environment 
# prepare = PrepareData(timewindow=timewindow, step=step)
				
# # 	# split data
# X_train, X_test, y_train, y_test, y_final_test, X_final_test = prepare.get_train_test_split()
# X_train.to_csv(path + 'train_quote_feature.csv')
# # dummy_clf = DummyClassifier(strategy="stratified")
# dummy_clf.fit(X_test, y_test)
# y_pred = dummy_clf.predict(X_final_test)
# report = classification_report(y_final_test, y_pred, output_dict=True)
# print(report)
	# return report 
					
# report = dummy_classifier('30_days')

# {'0': {'precision': 0.4186046511627907, 'recall': 0.391304347826087, 'f1-score': 0.4044943820224719, 'support': 46}, '1': {'precision': 0.40425531914893614, 'recall': 0.4318181818181818, 'f1-score': 0.41758241758241754, 'support': 44}, 'accuracy': 0.4111111111111111, 'macro avg': {'precision': 0.41142998515586343, 'recall': 0.4115612648221344, 'f1-score': 0.4110383998024447, 'support': 90}, 'weighted avg': {'precision': 0.41158942217823963, 'recall': 0.4111111111111111, 'f1-score': 0.41089297718511203, 'support': 90}}





#for debug 
# prepare = PrepareData()
# #fea = prepare.merge_data()
# X_train, X_test, y_train, y_test, y_final_test, X_final_test = prepare.get_train_test_split()
# experiment = load_experiment(prepare.path + '../experiment/experiment.yaml')
# parameters = experiment['experiment']['sklearn.linear_model.LogisticRegression']
# features_list = experiment['features']['set5']
# tfidf_words = 2000


# training = TrainingClassifiers(X_train = X_train, X_test =X_test, y_train = y_train, y_test =y_test, parameters=parameters, features_list=features_list, tfidf_words= tfidf_words)

# report, grid_search, pipeline = training.test_model(prepare.path, 'sklearn.linear_model.LogisticRegression')

#X_train = X_train, X_test =X_test, y_train = y_train, y_test =y_test, parameters=parameters, features_list=features_list, tfidf_words= tfidf_words

# fea_list = []
# for fea in features_list: 
# 	f_list = [i for i in X_train.columns if fea in i]
# 	fea_list.append(f_list)
# #flatten a list
# flat = [x for sublist in fea_list for x in sublist]
# #convert to transformer object
# selected_features = FunctionTransformer(lambda x: x[flat], validate=False)

#pipeline = training.setup_pipeline(features_list, 'sklearn.ensemble.RandomForestClassifier', tfidf_words)
#grid_search = training.training_models(pipeline)
#training.test_model(features_list, 'sklearn.ensemble.RandomForestClassifier', tfidf_words)


# c = Count_Vect()
# text = prepare.liwc_preprocess(365)
# # text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
# # text['text'] = text['text'].apply(lambda x: c.lemmatization(x))
# # text = c.parallelize_dataframe(text, c.get_precocessed_text)
# text.to_csv(path + 'process.csv')


# t = LDATopicModel()
# ldaText = prepare.liwc_preprocess(365)
# ldaText.to_csv(prepare.path + 'sample.csv')
# # lda_score_df, lda_model = t.get_lda_score(ldaText, 30)

