from transformers import pipeline
import pandas as pd
from scipy import stats
import statistics

#read data
class GetSentiment:
    """Here we use pretrained Roberta to predict sentiment """

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.file = pd.read_csv(self.path + "all_labels_reference.csv", encoding='latin-1') # file name
    
    def read_text(self):
        """Read text to dictionary."""

        text_dict = {}

        for text, textID in zip(self.file['text'], self.file['textID']):
            text_dict[textID] = text
        
        return text_dict

    def get_sentiment(self):
        """Transformer model predict sentiment.""" 
        # need to change the condition according to the structure of model result

        classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        text_dict = self.read_text()

        trans_sent = {}
        for k, v in text_dict.items():

            v = v.replace('**bobsnewline**', '')

            #bert cant't handle string longer than 512
            if len(v.split()) > 200: 
                select = v.split()[0:200]
                shortened = ' '.join(select)
                sent = classifier(shortened)

                if sent[0]['label'] == 'NEGATIVE':
                    score = sent[0]['score'] * -1
                    trans_sent[k] = score

                # elif sent[0]['label'] == '3 stars': 
                #     score = 0
                #     trans_sent[k] = score

                else:
                    trans_sent[k] = sent[0]['score']


            else:
                sent = classifier(v)
                print(k)
                print(v)

                if sent[0]['label'] == 'NEGATIVE':
                    score = sent[0]['score'] * -1
                    trans_sent[k] = score

                # elif sent[0]['label'] == '3 stars':
                #     score = 0
                #     trans_sent[k] = score

                else:
                    trans_sent[k] = sent[0]['score']
            print(trans_sent[k])
        return trans_sent


    def sentiment_df(self):
        """Convert sentiment dictionary to df.""" 

        trans_sent = self.get_sentiment()
        sent_df = pd.DataFrame.from_dict(trans_sent, orient='index')
        sent_df['textID'] = sent_df.index
        sent_df.columns = ['trans_sent', 'textID']

        return sent_df


class Stats:

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.nlptown = pd.read_csv(self.path + "transformers_sentiment_nlptown.csv") # file name
        self.cardiff = pd.read_csv(self.path + "transformers_sentiment_cardiff.csv")
        self.distilbert = pd.read_csv(self.path + "transformers_sentiment_distilbert.csv")
        self.all_var = pd.read_csv(self.path + "all_labels_reference.csv", encoding='latin-1')
        self.ensemble_score = pd.read_csv(self.path + "ensemble_score.csv")
        self.all_senti = pd.merge(self.ensemble_score, self.all_var, on='textID')


    def senti_cor(self):
        """Get the sentiment between old and new sentiment."""
        self.all_senti.to_csv(self.path + 'all_senti.csv')

        result = stats.pearsonr(self.all_senti['ensemble'], self.all_senti['sentiment_sum'])


        return result

    def lyrics_sentiment_by_group(self):
        """Compare lyric sentiment in high low group """

        lyrics = self.all_senti[self.all_senti['label'] == 2]

        high = lyrics[lyrics['cesd_sum'] >= 23]
        low = lyrics[lyrics['cesd_sum'] < 23]

        h_senti_m = statistics.mean(high['trans_sent'])
        l_senti_m = statistics.mean(low['trans_sent'])

        t_test = stats.ttest_ind(high['trans_sent'], low['trans_sent'])
        
        return h_senti_m, l_senti_m, t_test

    def ensemble_score(self):
        """Get the averaged score."""

        all_sentiment = pd.merge(self.nlptown, self.cardiff, on='textID')
        all_sentiment = pd.merge(all_sentiment, self.distilbert, on='textID')
        all_sentiment['ensemble'] = (all_sentiment['trans_sent'] + all_sentiment['trans_sent_x'] + all_sentiment['trans_sent_y'] )/3

        all_sentiment.to_csv(self.path + 'ensemble_score.csv')

        return all_sentiment
        


sent = GetSentiment()
#get sentiment file
#trans_sent = sent.sentiment_df()
#trans_sent.to_csv(sent.path + "transformers_sentiment_distilbert.csv")


# get correlation between old and new sentiment
com_stats = Stats()

# get ensemble score
#ensemble_score = com_stats.ensemble_score()
cor_result = com_stats.senti_cor()


# # #label 2 lyrics, label 1 quote
# lh_senti_m, ll_senti_m, qt_test = com_stats.lyrics_sentiment_by_group()
# qh_senti_m, ql_senti_m, qt_test = com_stats.quote_sentiment_by_group()



#ensembling 
#(0.0037779542938983597, 0.3984582619647376)

#cardiffnlp/twitter-roberta-base-sentiment   #This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark.
#(0.004021371719720609, 0.3687685434296002)


#nlptown/bert-base-multilingual-uncased-sentiment
#This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above, or for further finetuning on related sentiment analysis tasks.
#(0.005514401495445692, 0.21776938963186335)

#By default, the model downloaded for this pipeline is called “distilbert-base-uncased-finetuned-sst-2-english”. We can look at its model page to get more information about it. It uses the DistilBERT architecture and has been fine-tuned on a dataset called SST-2 for the sentiment analysis task.
#(0.00154571169575517, 0.7297428379360629)


#DistilBERT and nlptown (0.5866713864723485, 0.0)
#distilbert and cardiff ( (0.5722916008818502, 0.0))
#cardiff and nlptown (0.6138756979956137, 0.0)







