from transformers import pipeline
import pandas as pd
from scipy import stats

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

        classifier = pipeline('sentiment-analysis', model = 'm3hrdadfi/albert-fa-base-v2-sentiment-binary')
        text_dict = self.read_text()

        trans_sent = {}
        for k, v in text_dict.items():

            #bert cant't handle string longer than 512
            if len(v.split()) > 100: 
                select = v.split()[0:100]
                shortened = ' '.join(select)
                sent = classifier(shortened)

                if sent[0]['label'] == 'Negative':
                    score = sent[0]['score'] * -1
                    trans_sent[k] = score

                else:
                    trans_sent[k] = sent[0]['score']


            else:
                sent = classifier(v)
                print(k)

                if sent[0]['label'] == 'Negative':
                    score = sent[0]['score'] * -1
                    trans_sent[k] = score

                else:
                    trans_sent[k] = sent[0]['score']

        return trans_sent


    def sentiment_df(self):
        """Convert sentiment dictionary to df.""" 

        trans_sent = self.get_sentiment()
        sent_df = pd.DataFrame.from_dict(trans_sent, orient='index')
        sent_df['textID'] = sent_df.index
        sent_df.columns = ['trans_sent', 'textID'] 

        return sent_df


class Stats:

    def __init__(self, sentiment_df):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.senti = pd.read_csv(self.path + "transformers_sentiment.csv") # file name
        self.all_var = pd.read_csv(self.path + "all_labels_reference.csv", encoding='latin-1')
        self.all_var = pd.merge(self.senti, self.all_var, on = 'textID')


    def senti_cor(self):
        """Get the sentiment between old and new sentiment."""

        result = pearsonr(self.all_var['trans_sent'], self.all_var['sentiment_sum'])
        return result

    def lyrics_sentiment_by_group(self):
        """Compare lyric sentiment in high low group """

        lyrics = self.all_var[self.all_var['label'] == 2, ]

        high = lyrics[lyrics['cesd_sum'] > 23, ]
        low = lyrics[lyrics['cesd_sum'] < 23, ]

        h_senti_m = mean(high['trans_sent'])
        l_senti_m = mean(low['trans_sent'])

        t_test = stats.ttest_ind(h_senti_m, l_senti_m)
        
        return  h_senti_m, l_senti_m, t_test

    def quote_sentiment_by_group(self):


        lyrics = self.all_var[self.all_var['label'] == 1, ] #1 as quote

        high = lyrics[lyrics['cesd_sum'] > 23, ]
        low = lyrics[lyrics['cesd_sum'] < 23, ]

        h_senti_m = mean(high['trans_sent'])
        l_senti_m = mean(low['trans_sent'])

        t_test = stats.ttest_ind(h_senti_m, l_senti_m)
        
        return  h_senti_m, l_senti_m, t_test

        


sent = GetSentiment()
#get sentiment file
trans_sent = sent.sentiment_df()
trans_sent.to_csv(sent.path + "transformers_sentiment.csv")


#trans_sent = sent.read_text()
#v = trans_sent.get(90406)
#select = v.split()[0:100]
#shortened = ' '.join(select)
#sent = classifier(shortened)

# get correlation between old and new sentiment
#stats = Stats()
# cor_result = stats.senti_cor()

#label 2 lyrics, label 1 quote
#lh_senti_m, ll_senti_m, lt_test = stats.lyrics_sentiment_by_group()
#qh_senti_m, ql_senti_m, qt_test = stats.quote_sentiment_by_group()

#classifier = pipeline('sentiment-analysis', model = 'm3hrdadfi/albert-fa-base-v2-sentiment-binary')
# example
#classifier('Daddy, are you out there? Daddy, won\'t you come and play?')
#[{'label': 'NEGATIVE', 'score': 0.9663571119308472}]
