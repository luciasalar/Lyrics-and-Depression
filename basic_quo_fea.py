"""Here we compute the basic stats for user quotes."""
import pandas as pd
#from QuoteDetector import * 
class SelectText:
    def __init__(self):
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.participants = pd.read_csv(self.path + 'participants_matched.csv')
        self.frequent_users = pd.read_csv(self.path + 'adjusted_sample2.csv')
        self.sentiment_status = pd.read_csv(self.path + 'status_sentiment.csv')

    def selected_text(self):
        '''process participant files, select useful columns '''
        self.frequent_users.columns = ['rowname', 'userid', 'freq']
        participants = self.participants[['userid', 'time_completed', 'cesd_sum']]

        #frequent participants (commment this one if not needed)
        participants = pd.merge(self.frequent_users, participants, on='userid')
        participants.drop('rowname', axis=1, inplace=True)
        participants.drop('freq', axis=1, inplace=True)
        senti_part = pd.merge(self.participants, self.sentiment_status, on='userid')
        senti_part = senti_part[['userid', 'text']]
        senti_part['textID'] = senti_part.index
        return senti_part

    def selected_text2(self):
        '''process participant files, select useful columns '''
        self.frequent_users.columns = ['rowname', 'userid', 'freq']
        participants = self.participants[['userid', 'time_completed', 'cesd_sum']]

        #frequent participants (commment this one if not needed)
        participants = pd.merge(self.frequent_users, participants, on='userid')
        participants.drop('rowname', axis=1, inplace=True)
        participants.drop('freq', axis=1, inplace=True)
        senti_part = pd.merge(self.participants, self.sentiment_status, on='userid')
        senti_part = senti_part[['userid', 'text', 'positive', 'negative', 'time']]
        senti_part['textID'] = senti_part.index
        return senti_part


class CountQuote:
    """Compute stats."""

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.sp = SelectText()

    def recode(self):
        """Recode labels."""
        #get label file
        quote_or_lyrics = pd.read_csv(self.path + 'quote_all_labels.csv')
        quote_or_lyrics = quote_or_lyrics[['textID', 'label']]
        #labels = label1.merge(quote_or_lyrics)
        #labels['label'] = labels['label'].replace(['NonQuote','Quote','quote'], [0, 1, 1])
        quote_or_lyrics['label'] = quote_or_lyrics['label'].replace(['NonQuote','lyrics','quote'], [0, 2, 1])
        return quote_or_lyrics[['textID', 'label']]

    def merge_participants(self):
        """get sum sentiment score for each post"""
        participants = self.sp.selected_text2()
        participants['sentiment_sum'] = participants['positive'] + participants['negative']

        # recoded labels
        labels = self.recode()
        # merge
        par_label = participants.merge(labels, on='textID', how='left')
        # replace nan with 0
        par_label = par_label.fillna(0)
        return par_label

    def quotes_per_user(self):
        """Here we count the quotes in each user."""
        all_data = self.merge_participants()
        per_user = all_data.groupby(['userid', 'label']).size().to_frame(name='counts').reset_index()
        return per_user

    def count_quote(self):
        """Here return the number of quotes per users."""
        per_user = self.quotes_per_user()
        quotes_c = per_user.loc[per_user['label'] == 1]
        return quotes_c

    def count_quote_sentiment(self):
        """Here counts the quotes with different sentiments."""

        # count the number of posts in total
        all_data = self.merge_participants()
        all_count = all_data.groupby(['userid']).size().to_frame(name='all_count').reset_index()
        # get positive
        positive = all_data.loc[all_data['sentiment'] > 0]
        per_user = positive.groupby(['userid', 'label']).size().to_frame(name='pos_counts').reset_index()
        positive_c = per_user.loc[per_user['label'] == 1]
        # negative
        negative = all_data.loc[all_data['sentiment'] < 0]
        per_user2 = negative.groupby(['userid', 'label']).size().to_frame(name='nega_counts').reset_index()
        negative_c = per_user2.loc[per_user2['label'] == 1]
        # neutral
        neutral = all_data.loc[all_data['sentiment'] == 0]
        per_user3 = neutral.groupby(['userid', 'label']).size().to_frame(name='neu_counts').reset_index()
        neutral_c = per_user3.loc[per_user3['label'] == 1]
        return positive_c, negative_c, neutral_c, all_count

class QuoteFeatures:

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.countQuote = CountQuote()
    
    def get_count_quote(self):
        '''Here we see the count of valenced post in each user'''
        p, nega, neu, all_count = self.countQuote.count_quote_sentiment()
         # merge all the counts as feature '''
        quotation_fea = p.merge(nega, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(neu, on='userid', how='outer')
        quotation_fea = quotation_fea.merge(all_count, on='userid', how='outer')
        quotation_fea = quotation_fea[['userid', 'pos_counts', 'nega_counts', 'neu_counts','all_count']]
        quotation_fea = quotation_fea.fillna(0)
        quotation_fea['pos_counts'] = quotation_fea['pos_counts'] / quotation_fea['all_count']
        quotation_fea['nega_counts'] = quotation_fea['nega_counts'] / quotation_fea['all_count']
        quotation_fea['neu_counts'] = quotation_fea['neu_counts'] / quotation_fea['all_count']
        return quotation_fea

    def get_quote_senti(self):
        """get sentiment score """
        data = self.countQuote.merge_participants()
    
        quote_label = []
        for senti, quote in zip(data['sentiment_sum'], data['label']):
            if quote == 1 or quote == 2:
                quote_label.append(senti)

            else:
                quote_label.append(quote)

        data['quote_senti'] = pd.Series((v for v in quote_label))
        data.to_csv(self.path + 'quote_fea.csv')

        return data

# q = QuoteFeatures()
# fea = q.get_quote_senti()

# c = CountQuote()
# post = c.merge_participants()
# lyrics = post.loc[post['tag'] == 2]

q = QuoteFeatures()
c = CountQuote()
re = c.recode()
fea = q.get_quote_senti()


if __name__ == "__main__":

   q = QuoteFeatures()
   fea = q.get_quote_senti()
# count_fea = q.get_count_quote()
# t = SelectText()
# text = t.selected_text2()
# q = CountQuote()
# p = q.merge_participants()

#f.loc[f['label'] == 0]

