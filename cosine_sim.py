from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from CountVect import * 
from QuoteNDepStats import *

# path = get_tmpfile("word2vec.model")

# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
# model.train([["hello", "world"]], total_examples=1, epochs=1)


# sentences = [['first', 'sentence'], ['second', 'sentence'], ['hello', 'world']]
# train word2vec on the two sentences

def get_lyrics():
		"""retrieve data"""
		s = stats(180)
		post = s.get_data_by_day()
		nonOrg = post.loc[post['label'] == 1]
		lyrics = nonOrg.loc[nonOrg['tag'] == 2]

		return lyrics

def get_quotes():
	"""retrieve data"""
	s = stats(180)
	post = s.get_data_by_day()
	nonOrg = post.loc[post['label'] == 1]
	quote = nonOrg.loc[nonOrg['tag'] == 1]

	return quote

def get_non_orginal():
	"""retrieve data"""
	s = stats(180)
	post = s.get_data_by_day()
	nonOrg = post.loc[post['label'] == 0]
	return nonOrg


class word2Vec:
	def __init__(self, data):
		self.text = data
		

	def clean_text(self):
		"""clean text and remove noise"""

		c = Count_Vect()
		text = self.text[['text','userid']]
		text['text'] = text['text'].apply(lambda x: c.remove_noise(str(x)))
		text= c.get_precocessed_text(text)

		text['text'] = text['text'].apply(lambda x: x.split())

		return text

	def train_model(self, clean_text):
		"""train model and return word embeddings"""


		model = Word2Vec(clean_text['text'], min_count=1)
		word_vectors = model.wv

		return word_vectors

	def get_top_similar(self, keyword):
		"""get the top similar words"""
		clean_text = self.clean_text()
		word_vectors = self.train_model(clean_text)
		result = word_vectors.most_similar(positive=[keyword], topn=20)
		print(result)

		return result

# search I, you, love, want
lyrics = get_lyrics()
w = word2Vec(data=lyrics)
result1 = w.get_top_similar('want')


quotes = get_quotes()
w = word2Vec(data=quotes)
result2 = w.get_top_similar('want')

nonOrg = get_non_orginal()
w = word2Vec(data=nonOrg)
result3 = w.get_top_similar('love')