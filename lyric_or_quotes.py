"""This script annotate whether a post is a quote or lyrics."""
#from QuoteDetector import *
import re
import csv
import spacy
import gc
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from ruamel import yaml
import datetime
import os

"""After we non-original content annotation, this script classify whether the non-original content is lyrics """
def load_experiment(path_to_experiment):
    #load experiment 
    data = yaml.safe_load(open(path_to_experiment))
    return data

class MyQuote:
    """Create a class to store each variable as an object."""

    def __init__(self, text):
        """Read in CSV, process."""
        self.quoteText = text
        self.quoteID = hash(self.quoteText)
        self.textID = []
        self.link = []
        self.name = []
        self.description = []
        self.labels = []
        self.lyric_count = int()
        self.quote_count = int()
      

    def __hash__(self):
        """Hash with quote ID."""
        return self.quoteID

    def __str__(self):
        """Hash with quote ID."""
        return "Object text: " + self.quoteText + '\n' +\
               "Links: " + str(self.link) + '\n' +\
               "Names: " + str(self.name) + '\n' +\
               "Description " + str(self.description) + '\n' +\
               "labels: " + str(self.labels)  
               # "tag_count: " + str(self.tag_count) 


class GetLabels:
    """Get quotation labels."""

    def __init__(self, filename, feature_file):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.search_file = filename # search file name
        self.feature_file = feature_file


    def preprocess_fbp(self, sent):
        """Preprocess text."""
        sent = sent.replace('**bobsnewline**', '')
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)

            new_words.append(w)
        return ' '.join(new_words)

    def preprocess(self, sent):
        """Preprocess text."""
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)
            new_words.append(w)
        return ' '.join(new_words)

    def select_quotes(self):
        """Here we select the non-orginal content in all search."""
        file = pd.read_csv(self.path + 'quoteLabel_all.csv')
        non_orginal = file.loc[file['label'] == 'quote']
        non_orginal = non_orginal.drop(['text'], axis=1)
        # read all search
        all_search = pd.read_csv(self.path + 'all_search.csv')
        # select non original content
        non_orginal_search = all_search.merge(non_orginal, on='textID', how='inner')
        non_orginal_search.to_csv(self.path + 'all_search/non_orginal_search.csv')
        return non_orginal_search

    def hash_results(self, searchfilename):
        """Hash label result as dictionary. This is to count the keywords in website name"""
        objects = {}
        with open(self.path + searchfilename, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                texthash = hash(row['text'])
                if texthash not in objects:
                    objects[texthash] = MyQuote(row['text'])
                objects[texthash].link.append(row['link'])
                objects[texthash].name.append(row['name'])
                objects[texthash].description.append(row['description'])
                objects[texthash].textID.append(row['textID'])
                line = self.preprocess(row['name'])
                # count keywords
                count_lyrics = line.count("lyrics")
                count_quotes = line.count("quote")
                # check how many times keywords appear in names
                if count_lyrics >= 1:
                    objects[texthash].labels.append('lyrics')
                # count quotes
                else:
                    objects[texthash].labels.append('quotes')

                # else:
                #     objects[texthash].labels.append('other')
                
                count_lyrics = 0
                count_quote = 0
                for i in objects[texthash].labels:
                    if i == 'lyrics':
                        count_lyrics = count_lyrics + 1
                    else:
                        count_quote = count_quote + 1

                if count_lyrics >= count_quote:
                    objects[texthash].lyric_count = count_lyrics
                else:
                    objects[texthash].quote_count = count_quote

        return objects


    def get_quote_label(self, user_dict):
        """Here we store the user object 'quote_or_lyrics.csv' from the dictionary, 
        df contain the tag counts """ 

        f = open(self.path + self.feature_file, 'w')
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["textID"] + ["text"] + ["all_labels"] + ['lyric_count'] + ['quote_count'])

        for item in user_dict:
             writer.writerow([user_dict[item].textID[0]] + [user_dict[item].quoteText] + [user_dict[item].labels] + [user_dict[item].lyric_count]+ [user_dict[item].quote_count])
        f.close()

        qol = pd.read_csv(self.path + self.feature_file)
        return qol

    def merge_counts(self):
        #     """Merge keyword counts with cosine similarity. """
        counts = pd.read_csv(self.path + self.feature_file)
        cosine_sim = pd.read_csv(self.path + 'quoteLabel_all.csv')
        cosine_sim = cosine_sim[['textID', 'cosineSim_en', 'cosineSim_lg']]
        lyrics_fea = counts.merge(cosine_sim, on='textID', how='outer')
        print(lyrics_fea.shape)
        # merge with human labels 
        human_label = pd.read_csv(self.path + 'quote_or_lyrics_annotate2.csv', encoding = "ISO-8859-1", engine='python')
        print(human_label.shape)
        human_label = human_label[['textID', 'human']]
        lyrics_fea = lyrics_fea.merge(human_label, on='textID', how='inner')
        lyrics_fea.to_csv(self.path + 'lyrics_fea.csv')
        return lyrics_fea

    def merge_count2(self):
        #     """Merge keyword counts with cosine similarity. """
        counts = pd.read_csv(self.path + self.feature_file)
        cosine_sim = pd.read_csv(self.path + 'quoteLabel_all.csv')
        cosine_sim = cosine_sim[['textID', 'cosineSim_en', 'cosineSim_lg']]
        lyrics_fea = counts.merge(cosine_sim, on='textID', how='inner')
        return lyrics_fea

    def split_train_test(self):
        """Split train test """
        lyrics_fea = self.merge_counts()
        lyrics_fea = shuffle(lyrics_fea)

        quote = lyrics_fea.loc[lyrics_fea['human'] == 'quote']
        lyrics = lyrics_fea.loc[lyrics_fea['human'] == 'lyrics']
        NonQuote = lyrics_fea.loc[lyrics_fea['human'] == 'NonQuote']
      
        # get train set
        train1 = quote.head(205)
        train2 = lyrics.head(105)
        train3 = NonQuote.head(205)
        train = train1.append(train2)
        train = train.append(train3)
        # get test set
        test1 = quote.tail(89)
        test2 = lyrics.tail(45)
        test3 = NonQuote.head(89)
        test = test1.append(test2)
        test = test.append(test3)

        train = shuffle(train)
        test = shuffle(test)

        return train, test


    def annotate_quote_or_lyric(self, input_file, output_file, count, cos):
        """Annotate lyrics or quotes"""
        # merge with machine labels 
        machine_anno = pd.read_csv(self.path + 'quoteLabel_all.csv')
        machine_anno = machine_anno[['textID', 'label']]
        input_file = input_file.merge(machine_anno, on='textID')

        tag = []
        for c, cos, quote, label in zip(input_file.lyric_count, input_file.cosineSim_lg, input_file.quote_count, input_file.label):
            if (c > count) & (label == 'quote'):# 2
                tag.append('lyrics')
            elif (c > 0 & int(cos) > cos):# 0.98
                tag.append('lyrics')
            else:
                tag.append(label)

        input_file['machine'] = tag
        input_file.to_csv(self.path + output_file)
        return input_file

    def get_evaluation_train(self, result_df):
        """Get evaluation result."""
        #y_true_df = pd.read_csv(self.path + 'annotate_sample_train.csv')
        y_true_df = result_df[['textID', 'human']]
        y_pred_df = result_df[['textID', 'machine']]
        result_df = y_true_df.merge(y_pred_df, on='textID', how='inner')
        report = classification_report(result_df['human'], result_df['machine'], output_dict=True)
        return report

    def combine_all_labels(self):
        lyrics_lab = pd.read_csv(self.path + "lyrics_output_all.csv")
        lyrics_lab = lyrics_lab[['textID', 'machine']]
        quote_lab = pd.read_csv(self.path + 'quoteLabel_all_bleu.csv')
        all_lab = lyrics_lab.merge(quote_lab, on='textID', how='right')
        new_lab = []
        for lyric, label in zip(all_lab['machine'], all_lab['label']):
            if lyric == 'lyrics':
                new_lab.append('lyrics')
            else:
                new_lab.append(label)
        all_lab['label'] = new_lab
        all_lab.to_csv(self.path + 'quote_all_labels2.csv')
        return all_lab


def loop_da_grid():

    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression'
    experiment = load_experiment(path + '/experiment/experiment.yaml')

    l = GetLabels('all_search/train_search.csv', 'quote_or_lyrics.csv')
    # get search results and count keywords, then harsh to dict
    non_original_search = l.select_quotes()
    user_dict = l.hash_results('all_search/non_orginal_search.csv')
    # merge the counts with labels
    fea = l.merge_counts()# return feature file
    qol = l.get_quote_label(user_dict)
    train, test = l.split_train_test()
    print(train.shape)

    # stored result in file
    file_exists = os.path.isfile(l.path + '../result/lyrics_accuracy.csv')
    f = open(l.path + '../result/lyrics_accuracy.csv', 'a')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer_top.writerow(['count'] + ['cosine'] + ['train_result'] + ['test_result'] +['time'] )

    for count in experiment['lyrics']['count']:
        for cosine in experiment['lyrics']['cosine']:

            train_re = l.annotate_quote_or_lyric(train, 'lyrics_output_train.csv', count, cosine)
            report_train = l.get_evaluation_train(train_re)

            test_re = l.annotate_quote_or_lyric(test, 'lyrics_output_test.csv', count, cosine)
            report_test = l.get_evaluation_train(test_re)
            print(report_test)

            f = open(l.path + '../result/lyrics_accuracy.csv', 'a')
            result_row = [[count, cosine, pd.DataFrame(report_train), pd.DataFrame(report_test), str(datetime.datetime.now())]]

            writer_top.writerows(result_row)
            gc.collect()

            f.close()

loop_da_grid()
#l = GetLabels('all_search/all_search.csv', 'quote_or_lyrics.csv')
#new = l.combine_all_labels()

if __name__ == "__main__":
    # get individual model 
    l = GetLabels('all_search/all_search.csv', 'quote_or_lyrics.csv')
    non_original_search = l.select_quotes()
    user_dict = l.hash_results('all_search/non_orginal_search.csv')
    # merge the counts with labels
    fea = l.merge_count2()# return feature file
    all_quotes = l.get_quote_label(user_dict)
    labels = l.annotate_quote_or_lyric(fea, 'lyrics_output_all.csv', 3, 0.98)

    #get new labels



   







