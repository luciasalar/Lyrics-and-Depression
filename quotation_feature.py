import pandas as pd 
import datetime
import collections 
import numpy as np
import statistics
from basic_quo_fea import *
#other paths

#create dynamic quotation features for the classifier

class QuotationFeaDynamic:

	def __init__(self):
		self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/' 
		self.sentiment_status = pd.read_csv(self.path + 'status_sentiment.csv')
		self.participants = pd.read_csv(self.path + 'participants_matched.csv')
		self.sel_participants = pd.read_csv(self.path + 'only2011_users.csv')	
		self.qf = QuoteFeatures()
		self.quote_fea = self.qf.get_quote_senti()

	
	def get_relative_day(self, time_frame):
		'''this function returns date the post is written relatively to the day user complete the cesd '''
		#merge data with quotation features
		
		participants = pd.merge(self.participants, self.quote_fea, on = 'userid')
		participants = participants.drop(columns=['cesd_sum'])
		senti_part = participants.merge(self.sel_participants, on = 'userid', how = 'right')

		senti_part['time_diff'] = pd.to_datetime(senti_part['time_completed']) - pd.to_datetime(senti_part['time'])
		#select posts before completed cesd
		senti_part['time_diff'] = senti_part['time_diff'].dt.days
		senti_sel = senti_part[(senti_part['time_diff'] >= 0) & (senti_part['time_diff'] < time_frame)]
		print('there are {} posts posted before user completed cesd and within {} days'.format(senti_sel.shape[0], time_frame))
		return senti_sel


	def SortTime(self,file):
		'''sort post by time'''
		file = file.sort_values(by=['userid','time_diff'],  ascending=True)
		return file


	def user_obj(self, dfUserid, length):
		'''#create user vector of x(length) days'''
		users = {}
		for user in dfUserid:
		   # print(user)
		    if user not in users:
		        users[user] = [np.nan]*length
		return users


	def getAveragedValence(self, curdayPosts):
		'''get average mood of the day'''
		valence_sum = 0
		post_num = 1
		for post in curdayPosts:
		    valence_sum = valence_sum + post
		    post_num = post_num + 1

		valence_average = valence_sum/post_num
		return valence_average
		
	def getAveragedValenceVector(self, userObject, df):
		''' parse mood condition to a day framework, return a dictionary of user: day1 mood, day2 mood...'''
		preDay = np.nan
		preUser = np.nan
		preValence = np.nan
		curdayPosts = []
		i = 0
		for valence, day, user in zip(df['quote_senti'], df['time_diff'], df['userid']):
		    #rint(user, day)
		    #posVal = 0
		    if preUser is np.nan:
		        curdayPosts = [valence]
		    elif day == preDay and user == preUser:# and valence != preValence:
		        curdayPosts.append(valence)
		    else:
		        dayvalence = self.getAveragedValence(curdayPosts)
		        curdayPosts = [valence]
		        userObject[user][int(day)-1] = dayvalence
		    preDay = day
		    preUser = user
		    i +=1
		return userObject


	def get_mood_continous(self, timeRange):
		'''return a daily averaged mood  in certain time range '''
		#get sentiment sum
		# adjusted_sentiment = self.map_sentiment(self.sentiment_status)
		#get relative day
		sentiment_selected = self.get_relative_day(timeRange)
		#get mood score for each day , construct mood vector

		# sort posts according to time
		sorted_senti = self.SortTime(sentiment_selected)

		# get mood vector for X days
		users = self.user_obj(sorted_senti['userid'], timeRange)
		moodVector = self.getAveragedValenceVector(users, sorted_senti)
		return moodVector

	def get_mood_change_dict(self, timeRange, windowSzie, moodVector, step):
		'''construct mood temporal feature, how much does the mood changes every X day? X is the time window'''
		#count = 0
		mood_vect_window_dict= {}
		for k, v in moodVector.items():
			mood_vect_window  = []
			preWin_mean = 0 
			 
			for start in range(0,timeRange, step): #define data range
				end = start+windowSzie #window size
				#print(start, end)
				mini_vect = v[start:end]
				#mood_vect_window = []
				win_mean = np.nanmean(mini_vect)
				
				#get the change between two windows
				mood_vect_window.append(win_mean - preWin_mean)
				#print(win_mean)
				preWin_mean = win_mean

			mood_vect_window_dict[k] = mood_vect_window 
		return mood_vect_window_dict

	def get_mood_continous_dict(self, timeRange, windowSzie, moodVector, step):
		'''construct mood temporal feature, how much does the mood changes every X day? X is the time window'''
		#count = 0
		mood_vect_window_dict= {}
		for k, v in moodVector.items():
			mood_vect_window  = []
			preWin_mean = 0 
			 
			for start in range(0,timeRange, step): #define data range
				end = start+windowSzie #window size
				#print(start, end)
				mini_vect = v[start:end]
				#mood_vect_window = []
				win_mean = np.nanmean(mini_vect)
				#get the change between two windows
				mood_vect_window.append(win_mean)

			mood_vect_window_dict[k] = mood_vect_window 
		return mood_vect_window_dict


	def get_quote_change_in_timewindow(self, timeRange, windowSzie, step):
		# get mood vector with 
		moodVector = self.get_mood_continous(timeRange)
		mood_dict = self.get_mood_change_dict(timeRange, windowSzie, moodVector, step) #paramenter: number of days used as features, time window
		mood_vect_df = pd.DataFrame.from_dict(mood_dict).T

		mood_vect_df.to_csv(self.path + './mood_vectors/quote_change_frequent_user_window_{}_timeRange{}_step{}.csv'.format(windowSzie, timeRange, step)) #feature matrx for prediction 
		return mood_vect_df, windowSzie

	def get_quote_continous_in_timewindow(self, timeRange, windowSzie, step):
		# get mood vector with 
		moodVector = self.get_mood_continous(timeRange)
		mood_dict = self.get_mood_continous_dict(timeRange, windowSzie, moodVector, step) #paramenter: number of days used as features, time window
		mood_vect_con_df = pd.DataFrame.from_dict(mood_dict).T

		mood_vect_con_df.to_csv(self.path + './mood_vectors/quote_continous_user_window_{}_timeRange{}_step{}.csv'.format(windowSzie, timeRange, step)) #feature matrx for prediction 
		return mood_vect_con_df, windowSzie

if __name__ == "__main__":
#read sentiment file
# sp = SelectParticipants()
# path = sp.path
# participants = sp.process_participants()

#here you define the number of days you want to use as feature and the time window for mood
    quote = QuotationFeaDynamic()
# mood_vector_feature, windowSize = quote.get_quote_continous_in_timewindow(365, 14, 3)
#get mood feature in time window (category)
# mood_vector_feature, windowSzie = mood.get_mood_in_timewindow(365, 30, 3)

# #get mood feature in time window (continues)
# mood_vector_con_feature, windowSzie = mood.get_mood_continous_in_timewindow(365, 14, 3)

# mood_vector_feature, windowSize = quote.get_quote_change_in_timewindow(365, 14, 3)
