
import pandas as pd
from time import sleep
import csv
import pickle
import re
import spacy
import numpy as np
from sklearn.preprocessing import StandardScaler
#from googlesearch import search
from google import google
import random
import datetime

#coding: utf8
class SelectText:
    def __init__(self):
        self.path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/' 
        self.participants = pd.read_csv(self.path + 'participants_matched.csv')
        self.frequent_users = pd.read_csv(self.path + 'frequent_users.csv')
        self.sentiment_status = pd.read_csv(self.path + 'status_sentiment.csv')

    def selected_text(self):
        '''process participant files, select useful columns '''
        self.frequent_users.columns = ['rowname', 'userid','freq']
        participants  = self.participants[['userid','time_completed','cesd_sum']]

        #frequent participants (commment this one if not needed)
        participants = pd.merge(self.frequent_users, participants, on='userid')
        participants.drop('rowname', axis=1, inplace=True)
        participants.drop('freq', axis=1, inplace=True)
        senti_part = pd.merge(self.participants, self.sentiment_status, on = 'userid')
        senti_part = senti_part[['userid','text']]
        senti_part['textID'] = senti_part.index
        return senti_part

    def selected_text2(self):
        '''process participant files, select useful columns '''
        self.frequent_users.columns = ['rowname', 'userid','freq']
        participants  = self.participants[['userid','time_completed','cesd_sum']]

        #frequent participants (commment this one if not needed)
        participants = pd.merge(self.frequent_users, participants, on='userid')
        participants.drop('rowname', axis=1, inplace=True)
        participants.drop('freq', axis=1, inplace=True)
        senti_part = pd.merge(self.participants, self.sentiment_status, on = 'userid')
        senti_part = senti_part[['userid','text','positive','negative']]
        senti_part['textID'] = senti_part.index
        return senti_part


class SearchGoogle:

    def __init__(self, textFile, path):
        self.num_page = 1
        self.textFile = textFile
        self.path = path
        self.lineNum = 100000
  
    def search_google(self, SubtextFile):
        '''search function: This function return the First Page of the Google search result. The variables return include the link, name and content of the website'''
        counter = 0
        f = open(self.path + '/searchQuote/searchText{}.csv'.format(str(datetime.datetime.now())), 'w')
        writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)  
        writer.writerow(["textID"] + ["text"] + ["link"] + ["name"]+['description'])
        for textid,text in zip(SubtextFile['textID'], SubtextFile['text']):
            search_results = [[textid, text, i.link, i.name,i.description] for i in google.search(text, self.num_page)]        
            writer.writerows(search_results)
            sleep(random.randrange(0, 10, 1)) #sleep for random seconds

            counter = counter + 1
            if counter == self.lineNum:
                f.close()
                counter == 0
                #open another file
                f = open(self.path + '/searchQuote/searchText{}.csv'.format(str(datetime.datetime.now())), 'w')
                writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)  
                writer.writerow(["textID"] + ["text"] + ["link"] + ["name"]+['description'])
        

    def get_all_queries(self, startLine, endLine):
        start = startLine 
        for end in range(start, endLine, 500):
            textFile = self.textFile[start:end]
            #print(textFile)
            #s = SearchGoogle(path= sp.path, textFile = textFile)
            self.search_google(textFile)
            start = end
            
class myQuote:
    ''' create a class to store each variable as an object'''
    def __init__(self, text):
        #read in CSV, process
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
        return self.quoteID
    
    def __str__(self):
        return "Object text: " + self.quoteText + '\n' +\
        "Links: " + str(self.link) + '\n' +\
        "Names: " + str(self.name) + '\n' +\
        "Description " + str(self.description) + '\n' +\
        "Scores: " + str(self.scores)
        

class SearchFeatures:
    def __init__(self, path):
       
        self.path = path

    def preprocess(self, sent):
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)
            new_words.append(w)
            
        return ' '.join(new_words)

    def preprocess_fbp(self, sent):
        sent = sent.replace('**bobsnewline**', '')
        words = str(sent).lower().split()
        new_words = []
        for w in words:
            w = re.sub(r'[0-9]+', '', w)

            new_words.append(w)
            
        return ' '.join(new_words)
    
    def hash_results(self):
        objects = {}
        with open(self.path + 'searchQuote/searchText1.csv', 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
        #         print(row['text'])
                #print(row['text'])
                texthash = hash(row['text'])
                if texthash not in objects:
                    objects[texthash] = myQuote(row['text'])
                objects[texthash].link.append(row['link'])
                objects[texthash].name.append(row['name'])
                objects[texthash].description.append(row['description'])
                objects[texthash].textID.append(row['textID'])
                line = self.preprocess(row['name'])
                #count keywords
                count = line.count("lyric") + line.count("lyrics") + line.count("quote") + line.count("quotes")
                objects[texthash].scores.append(count)

        return objects


    def cosineSim(self):
        '''for each doc, compute cosine similarity between doc and returned text in dictionary, store result in dict'''
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


    
    def saveCosineSimAsCSV(self):
        '''here we save the file as csv, the csv file is just for your reference'''
        objects = self.cosineSim()
        f = open(self.path + 'searchQuote3.csv', 'w')
        writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)  
        writer.writerow(["textID"] + ["text"] + ["count"] + ["cosineSim"])
        for item in objects.keys():
            writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en] + [objects[item].cos_lg])
        f.close()

    def getQuoteLabel(self, filename):
        ''' Here we create a class to store each variable as an object
        Return the count of key words 'lyric, lyrics, quote, quotes' in the name of the website, because the name of the website contains most of the information we need
        Count the keywords and store it as Score. Score is a vector that contain keyword counts in each search result
        compute cosine similarity between post and retrieve website content'''

        objects = self.cosineSim()
        f = open(self.path + filename, 'w')
        writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)  
        writer.writerow(["textID"] + ["text"] + ["count"]+ ["cosineSim_en"] + ["cosineSim_lg"] + ['label'])
        for item in objects:
            if objects[item].cos_lg[0] > 0.988:
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['Quote'])
            # elif (objects[item].cos_lg[0] <= 0.988 and objects[item].cos_en[0] > 0.91):
            #     writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['Quote'])

            elif (objects[item].cos_lg[0] < 0.988 and objects[item].cos_lg[0] >= 0.975) or (objects[item].cos_lg[0] <= 0.988 and objects[item].cos_en[0] > 0.91):
                #check if title has keywords 
                count = 0
                for score in objects[item].scores:
                    if score > 0: # 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count > 2 :
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['quote'])

            elif (objects[item].cos_lg[0] < 0.975 and objects[item].cos_lg[0] > 0.90): 
                count = 0
                for score in objects[item].scores:
                    if score > 0: # 1 is 0 before we add 1 for smoothing
                        count = count + 1
                if count > 4 :
                    writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]]+  [objects[item].cos_lg[0]] + ['quote'])
                       
            else:
                writer.writerow([objects[item].textID[0]] + [objects[item].quoteText] + [objects[item].scores] + [objects[item].cos_en[0]] + [objects[item].cos_lg[0]] +['NonQuote'])
       # else:
           # writer.writerow([objects[item].quoteID] + [objects[item].quoteText] + [objects[item].scores] + ['null']+['NotQuote'])
        f.close()
    

if __name__ == '__main__':
    sp = SelectText()

    participants = sp.selected_text()


    s = SearchGoogle(path=sp.path, textFile=participants)
    s.get_all_queries(120001, 150000)

#get cosine similarity score, this table is used as feature directly
# search = SearchFeatures(path = sp.path) 
# #search.saveCosineSimAsCSV()
# search.getQuoteLabel('quoteLabel.csv')




# s = SearchObject(textFile = textFile)
# r = s.align_scores()


#search_results = search('this is my day', 1)
# ob = search_google_DF(participants['text'][:10])
# cosineSim(ob)
# align_scores(ob)


# proto_matrix = append_features(ob)
# FeatureMatrix = np.matrix(proto_matrix)
# scaler = StandardScaler()
# scaled_matrix = scaler.fit_transform(FeatureMatrix)
# scaled_matrix

# a = QuoteClassifier()
# a.isQuote("dasda")
