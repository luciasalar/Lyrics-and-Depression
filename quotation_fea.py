"""Here we compute the basic stats for user quotes."""
import pandas as pd
from QuoteDetector import * 

"""Compute statistics for the quotation feature """
class Stats:
    """Compute stats."""

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.sp = SelectText()

    def recode(self):
        """Recode labels."""
        labels = pd.read_csv(self.path + 'quoteLabel.csv')
        new_l = labels.replace(['NonQuote', 'Quote', 'quote'], [0, 1, 1])
        return new_l[['textID', 'label']]

    def merge_participants(self):
        """Merge labels with participants"""
        participants = self.sp.selected_text2()
        participants['sentiment'] = participants['positive'] + participants['negative']
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
        per_user = all_data.groupby(['userid', 'label']).size().to_frame(name='quote_count').reset_index()
        return per_user

    def count_quote(self):
        """Here return the number of quotes per users."""
        per_user = self.quotes_per_user()
        quotes_c = per_user.loc[per_user['label'] == 1]
        return quotes_c

    def count_quote_sentiment(self):
        """Her counts the quotes with different sentiments."""
        all_data = self.merge_participants()
        # get positive
        positive = all_data.loc[all_data['sentiment'] > 0]
        per_user = positive.groupby(['userid','label']).size().to_frame(name='quote_count_pos').reset_index()
        positive_c = per_user.loc[per_user['label'] == 1]
        # negative
        negative = all_data.loc[all_data['sentiment'] < 0]
        per_user2 = negative.groupby(['userid','label']).size().to_frame(name='quote_counts_neg').reset_index()
        negative_c = per_user2.loc[per_user2['label'] == 1]
        # neutral
        neutral = all_data.loc[all_data['sentiment'] == 0]
        per_user3 = neutral.groupby(['userid','label']).size().to_frame(name='quote_counts_neu').reset_index()
        neutral_c = per_user3.loc[per_user3['label'] == 1]
        return positive_c, negative_c, neutral_c


class QuoteFeature:
    """Create feature df."""
    def __init__(self):
        """Define varibles."""
        self.stats = Stats()
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'

    def merge_features(self):
        """create quote count feature matrix"""
        positive, negative, neutral = self.stats.count_quote_sentiment()
        all_q = self.stats.count_quote()
        # merge all features
        quote_fea = positive.merge(negative, on='userid', how='outer')
        quote_fea = quote_fea.merge(neutral, on='userid', how='outer')
        quote_fea = quote_fea.merge(all_q, on='userid', how='outer')

        # merge with participants
        participants = pd.read_csv(self.path + 'participants_matched.csv')
        participants  = participants[['userid','cesd_sum']]
        quote_fea = quote_fea.merge(participants, on='userid', how='outer')
        quote_fea = quote_fea.fillna(0)

        return quote_fea[['userid','quote_count_pos','quote_counts_neu','quote_counts_neg','quote_count']]
        #return quote_fea


q = QuoteFeature()
q_fea = q.merge_features()


