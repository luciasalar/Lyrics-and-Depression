"""This script annotate whether a post is a quote."""
#from QuoteDetector import *
import re
import csv
import spacy
import gc
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
recall_score, confusion_matrix, classification_report, accuracy_score 
from ruamel import yaml
import datetime
import os


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
        self.scores = []
        self.cos_en = []
        self.cos_lg = []

    def __hash__(self):
        """Hash with quote ID."""
        return self.quoteID

    def __str__(self):
        """Hash with quote ID."""
        return "Object text: " + self.quoteText + '\n' +\
               "Links: " + str(self.link) + '\n' +\
               "Names: " + str(self.name) + '\n' +\
               "Description " + str(self.description) + '\n' +\
               "Scores: " + str(self.scores)


class GetLabels:
    """Get quotation labels."""

    def __init__(self, filename):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.search_file = filename


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

    def hash_results(self):
        """Hash label result as dictionary."""
        objects = {}
        with open(self.path + self.search_file, 'r', encoding='utf-8-sig') as csvfile:
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
                count = line.count("lyric") + line.count("lyrics") + line.count("quote") + line.count("quotes") + line.count("proverbs")
                objects[texthash].scores.append(count)

        return objects

    def cosine_sim(self):
        """For each doc, compute cosine similarity between doc and returned text
        in dictionary, store result in dict."""
        nlplg = spacy.load('en_core_web_lg')
        nlp = spacy.load('en')
        results = self.hash_results()
        for item in results:
            str1 = ''.join(results[item].description)

            # preprocess documents
            pro_doc1 = self.preprocess(str1)
            pro_doc2 = self.preprocess_fbp(results[item].quoteText)
            # use spacy model on text
            doc2 = nlp(pro_doc2)
            doc1 = nlp(pro_doc1)
            doc2lg = nlplg(pro_doc2)
            doc1lg = nlplg(pro_doc1)

            # compute cosine similarity
            results[item].cos_en.append(doc1.similarity(doc2))
            results[item].cos_lg.append(doc1lg.similarity(doc2lg))
        return results

    def get_quote_label(self, filename, cos_lg1, cos_lg2, cos_lg3, cos_en, count1, count2):
        """Here we create a class to store each variable as an object
            Return the count of key words 'lyric, lyrics, quote, quotes' in the
            name of the website, because the name of the website contains most
            of the information we need
            Count the keywords and store it as Score. Score is a vector that
            contain keyword counts in each search result compute cosine
            similarity between post and retrieve website content"""

        objects = self.cosine_sim()
        f = open(self.path + filename, 'w')
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["textID"] + ["text"] + ["count"] + ["cosineSim_en"] +
        ["cosineSim_lg"] + ['label'])
        
        for item in objects:
           
           # if cosine similarity > 09=.9
            if objects[item].cos_lg[0] >= cos_lg1: #0.988
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] +
                [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] + ['quote'])
            # elif (objects[item].cos_lg[0] <= 0.988 and objects[item].cos_en[0] > 0.91):
            # writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['Quote'])

            elif (objects[item].cos_lg[0] < cos_lg1 and objects[item].cos_lg[0] >= cos_lg2) or (objects[item].cos_lg[0] < cos_lg1 and objects[item].cos_en[0] > cos_en):
                # check if title has keywords  cos_lg2= 0.975 cos_en = 0.75, cos_lg3= 0.85
                count = 0
                for score in objects[item].scores:
                    if score > 0:# 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count >= count1:
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['quote'])
                else:
                	writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] +['NonQuote'])

            elif (objects[item].cos_lg[0] < cos_lg2 and objects[item].cos_lg[0] > cos_lg3):
                count = 0
                for score in objects[item].scores:
                    if score > 0:# 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count > count2:
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] + ['quote'])
                else:
                	writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] +['NonQuote'])
            else:
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] +['NonQuote'])

       # else:


           # writer.writerow([objects[item].quoteID] + [objects[item].quoteText] + [objects[item].scores] + ['null']+['NotQuote'])
        f.close()
        return objects
            


# get cosine similarity score, this table is used as feature directly

    def get_train_set(self, line):
        """Select sample for processing."""
        # select the first 100 as sample
        sample = pd.read_csv(self.path + 'all_quoteLabel/quoteLabel3.csv')
        annotate_sample = sample.head(line)
        #annotate_sample.to_csv(self.path + 'annotate_sample_train.csv')
        # select annotated sample for automate an
        all_search = pd.read_csv(self.path + 'all_search/all_search3.csv')
        annotate_sample = annotate_sample.drop(['text'], axis=1)
        selected = all_search.merge(annotate_sample, on='textID', how='inner')
        selected.to_csv(self.path + 'all_search/train_search.csv')
        return selected

    def get_test_set(self, line):
        """Select sample for processing."""
        # select the first 100 as sample
        sample = pd.read_csv(self.path + 'all_quoteLabel/quoteLabel3.csv')
        annotate_sample = sample.tail(line)
        #annotate_sample.to_csv(self.path + 'annotate_sample_test.csv')
        # select annotated sample for automate an
        all_search = pd.read_csv(self.path + 'all_search/all_search3.csv')
        annotate_sample = annotate_sample.drop(['text'], axis=1)
        selected = all_search.merge(annotate_sample, on='textID', how='inner')
        selected.to_csv(self.path + 'all_search/test_search.csv')
        return selected


    def get_evaluation_train(self):
        """Get evaluation result."""
        y_true_df = pd.read_csv(self.path + 'annotate_sample_train.csv')
        y_true_df = y_true_df[['textID', 'human']]
        y_pred_df = pd.read_csv(self.path + 'quoteLabel_train.csv')
        y_pred_df = y_pred_df[['textID', 'label']]
        result_df = y_true_df.merge(y_pred_df, on='textID', how='inner')
        report = classification_report(result_df['human'], result_df['label'], output_dict=True)
        return report

    def get_evaluation_test(self):
        """Get evaluation result."""
        y_true_df = pd.read_csv(self.path + 'annotate_sample_test.csv')
        y_true_df = y_true_df[['textID', 'human']]
        y_pred_df = pd.read_csv(self.path + 'quoteLabel_test.csv')
        y_pred_df = y_pred_df[['textID', 'label']]
        result_df = y_true_df.merge(y_pred_df, on='textID', how='inner')
        report = classification_report(result_df['human'], result_df['label'], output_dict=True)
        return report


def loop_da_grid():
    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression'
    experiment = load_experiment(path + '/experiment/experiment.yaml')

    train = GetLabels('all_search/train_search.csv')
    test = GetLabels('all_search/test_search.csv')
    
    file_exists = os.path.isfile(train.path + '../result/quotation_accuracy.csv')
    f = open(train.path + '../result/quotation_accuracy.csv', 'a')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer_top.writerow(['cos_lg1'] + ['cos_lg2'] + ['cos_lg3'] + ['cos_en'] + ['train_report'] + ['test_report'] +['time'] + ['count1'] + ['count2'])

    for cos_lg1 in experiment['quote']['cos_lg1']:
        for cos_lg2 in experiment['quote']['cos_lg2']:
            for cos_lg3 in experiment['quote']['cos_lg3']:
                for cos_en in experiment['quote']['cos_en']:
                    for count1 in experiment['quote']['count1']:
                        for count2 in experiment['quote']['count2']:
                            p_obj = train.get_quote_label('quoteLabel_train.csv', cos_lg1, cos_lg2, cos_lg3, cos_en, count1, count2)
                            p_obj2 = test.get_quote_label('quoteLabel_test.csv', cos_lg1, cos_lg2, cos_lg3, cos_en, count1, count2)

                            report_train = train.get_evaluation_train()
                            report_test = test.get_evaluation_test()

                            f = open(train.path + '../result/quotation_accuracy.csv', 'a')
                            result_row = [[cos_lg1, cos_lg2, cos_lg3, cos_en, pd.DataFrame(report_train), pd.DataFrame(report_test), str(datetime.datetime.now()), count1, count2]]

                            writer_top.writerows(result_row)                    
                            gc.collect()
                            
                            f.close()

#loop_da_grid()
#loop_da_grid()

if __name__ == "__main__":
    #training 
    #g = GetLabels('all_search/train_search.csv')
    #test = g.get_test_set()
    loop_da_grid()

    g = GetLabels('all_search/all_search.csv')
    g.get_quote_label('quoteLabel_all.csv', 0.998, 0.975, 0.85, 0.75, 3, 4)









# count = 0
# for k, v in p_obj.items():
# 	print(v)

# dict_p = pd.DataFrame.from_dict(p_obj,  orient='index')