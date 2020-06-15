#import os
import pandas as pd
import numpy as np
import ibm_db
import ibm_db_dbi # part of ibm_db - no need to install separately 
#from keras.preprocessing.text import Tokenizer #word tokenization
from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing import sequence
#from tensorflow.keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers.embeddings import Embedding
np.random.seed(4064)
#import pandasql
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import pickle

#import time
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import keras
#from keras.initializers import Constant
from keras.models import model_from_json
#from keras import backend as K

#from numpy import array, argmax
#import tensorflow as tf

#import subprocess
#import sys
#import jhashcode

# ElasticSearch:
#from elasticsearch import RequestsHttpConnection, Elasticsearch
#import logging

class verity_db_connect:
    
    def connect(self):  ## db connection
        try:
			#Db details to be changed here
            dsn = ("DRIVER={{IBM DB2 ODBC DRIVER}};DATABASE=NPLDB;HOSTNAME=169.61.12.224;PORT=50000;PROTOCOL=TCPIP;UID=vapp;PWD=pwd4Verity@app")
            conn = ibm_db.connect(dsn, "", "")
            print("Connection successful")
        except:
            print("Connection Failed")
        return conn
    
class verity_sql:
    
    def __init__(self, schema_name):
        cnnOb = verity_db_connect()
        self.cnn=cnnOb.connect()
        self.conn=ibm_db_dbi.Connection(self.cnn)
        self.schema=schema_name
    
    def fetch_statements(self):
        sql = "SELECT ID, STATEMENT FROM "+self.schema+".STATEMENT where id > COALESCE((SELECT MAX(STATEMENT_ID) FROM "+self.schema+".STATEMENT_RATING_HISTORY),0)"
        return pd.read_sql(sql, self.conn)
    
    def fetch_iteration(self):
        itr = "SELECT COALESCE(MAX(ITERATION_ID),0) AS ITR FROM "+self.schema+".STATEMENT_RATING_HISTORY"
        return pd.read_sql(itr, self.conn)
    
    def update_statement(self,crnt_itr):
		#query = "UPDATE "+self.schema+".STATEMENT_RATING_HISTORY A SET A.RATING = CAST(ROUND(A.RATING, 1) AS DECIMAL(3,1)) WHERE A.ITERATION_ID = "+str(crnt_itr)
        query = "UPDATE "+self.schema+".STATEMENT_RATING_HISTORY A SET A.RATING = CAST(ROUND(A.RATING, 1) AS DECIMAL(3,1)) WHERE A.ITERATION_ID = "+str(crnt_itr)
        stmt = ibm_db.exec_immediate(self.cnn, query)
        print("Number of affected rows in STATEMENT_RATING_HISTORY: ", ibm_db.num_rows(stmt))
	
        query = "UPDATE "+self.schema+".STATEMENT A SET (A.RATING, A.FACT_FG) = (SELECT B.RATING, (CASE WHEN B.RATING >=0.5 THEN 1 ELSE 0 END) FROM "+self.schema+".STATEMENT_RATING_HISTORY AS B WHERE B.STATEMENT_ID = A.ID AND B.ITERATION_ID = "+str(crnt_itr)+") WHERE A.ID IN (SELECT B.STATEMENT_ID FROM "+self.schema+".STATEMENT_RATING_HISTORY B WHERE B.ITERATION_ID = "+str(crnt_itr)+")"
        stmt = ibm_db.exec_immediate(self.cnn, query)
        print("Number of affected rows in STATEMENT: ", ibm_db.num_rows(stmt))
        
    def insert_statement(self,data):
        placeholders = ', '.join(['?'] * len(data.columns))
        tuple_of_tuples = tuple([tuple(x) for x in data.values])

        insert_sql = "INSERT INTO "+self.schema+".STATEMENT_RATING_HISTORY(STATEMENT_ID, RATING, ITERATION_ID) VALUES ( " + placeholders + ")"
        stmt_insert = ibm_db.prepare(self.cnn, insert_sql)
        print("Number of rows inserted: ",ibm_db.execute_many(stmt_insert, tuple_of_tuples)) # load records
        
class cleanStatement:
    
   
    def text_to_wordlist(self, text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.
        
        # Convert words to lower case
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=$%]", " ", str(text))
        text = re.sub(r"what's", "what is", str(text))
        text = re.sub(r"\'s", " ", str(text))
        text = re.sub(r"\'ve", " have ", str(text))
        text = re.sub(r"can't", "cannot ", str(text))
        text = re.sub(r"n't", " not ", str(text))
        text = re.sub(r"dont", "do not ", str(text))
        text = re.sub(r"i'm", "i am ", str(text))
        text = re.sub(r"\'re", " are ", str(text))
        text = re.sub(r"\'d", " would ", str(text))
        text = re.sub(r"\'ll", " will ", str(text))
        text = re.sub(r",", " ", str(text))
        text = re.sub(r"\.", " ", str(text))
        text = re.sub(r"!", " ! ", str(text))
        text = re.sub(r"\/", " ", str(text))
        text = re.sub(r"\^", " ^ ", str(text))
        text = re.sub(r"\+", " + ", str(text))
        text = re.sub(r"\-", " - ", str(text))
        text = re.sub(r"\=", " = ", str(text))
        text = re.sub(r"'", " ", str(text))
        text = re.sub(r":", " : ", str(text))
        text = re.sub(r" e g ", " eg ", str(text))
        text = re.sub(r" b g ", " bg ", str(text))
        text = re.sub(r" u s ", " american ", str(text))
        text = re.sub(r" 9 11 ", "911", str(text))
        text = re.sub(r"e - mail", "email", str(text))
        text = re.sub(r"e-mail", "email", str(text))
        text = re.sub(r" j k ", " jk ", str(text))

        text = re.sub(r"\0s", "0", str(text))
        text = re.sub(r"(\d+)(k)", r"\g<1>000", str(text))
        text = re.sub(r"\s{2,}", " ", str(text))

        text = re.sub(r"\%", " percent", str(text))
        text = re.sub(r"\$(\d+)", r"\1 dollars", text)
      #  text = re.sub(r"\$", " dollars ", str(text))



        # Convert words to lower case and split them

        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split(" ")
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split(" ")
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return(text)

    
class modelScore:
    
    def execModel(self,dfCleanedDataSen):
        # loading tokenizer
        with open('Verity Tokens.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        comment_len = 100
        #pad the sequence for same length
        X_data = pad_sequences(tokenizer.texts_to_sequences(dfCleanedDataSen), padding='post', maxlen = comment_len)
        
        # load json and create model
        json_file = open('Modelp1e.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("Modelpe1.h5")
        print("Loaded model from disk")

        lcenter = np.load('Verity Centers.npy').T[0]
        cents = len(lcenter)
        eyes = np.eye(cents)
        cDiff = eyes[1:cents,:]-eyes[:(cents-1),:]

        score_out = np.array(model.predict(X_data)).reshape(-1)
        print("Data Scored")

        gaps = np.matmul(cDiff, lcenter)

        score_out = score_out.reshape(-1)
        score = score_out.copy()

        score[score_out>=lcenter[5]] = 5.0
        score[(score_out>=lcenter[4]) & (score_out<lcenter[5])] = (4.0 + (score_out[(score_out>=lcenter[4]) & (score_out<lcenter[5])] - lcenter[4])/gaps[4])
        score[(score_out>=lcenter[3]) & (score_out<lcenter[4])] = (3.0 + (score_out[(score_out>=lcenter[3]) & (score_out<lcenter[4])] - lcenter[3])/gaps[3])
        score[(score_out>=lcenter[2]) & (score_out<lcenter[3])] = (2.0 + (score_out[(score_out>=lcenter[2]) & (score_out<lcenter[3])] - lcenter[2])/gaps[2])
        score[(score_out>=lcenter[1]) & (score_out<lcenter[2])] = (1.0 + (score_out[(score_out>=lcenter[1]) & (score_out<lcenter[2])] - lcenter[1])/gaps[1])
        score[(score_out>=lcenter[0]) & (score_out<lcenter[1])] = (0.0 + (score_out[(score_out>=lcenter[0]) & (score_out<lcenter[1])] - lcenter[0])/gaps[0])
        score[score_out<lcenter[0]] = 0.0

        score = np.around(score, decimals=1)

        print("Loaded model from disk")
        print("Data Scored")

        scr = np.around(score, decimals=0)
        vals, counts = np.unique(scr, return_counts=True)
        print(np.concatenate((vals.reshape(-1,1),(counts/len(score)).reshape(-1,1)), axis=1))
        return score
        
    
def main():
    #Create a db connection object
    cnnOb = verity_db_connect();
    cnn=cnnOb.connect()
    cnn
    conn=ibm_db_dbi.Connection(cnn)
    conn
    
    #using the connection fetch the statements from database
	#schema name has to be changed
    sqlObj = verity_sql('VERITY_PRD_202005181200')
    curItr=sqlObj.fetch_iteration()
    dfStatements = sqlObj.fetch_statements()
    #dfStatements[:2]
    
    #Clean/standardize the statements
    objCleanStatements = cleanStatement()
    dfCleanedData = dfStatements.copy()
    dfCleanedData['STATEMENT'] = dfCleanedData['STATEMENT'].apply(lambda x: str(x).lower()) #lower case of the word
    #dfCleanedData['Sentence'] = cleanedData['Sentence'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]', ' ', x)))         #remove anything that is not a number or alphabet
    dfCleanedDataSen = dfCleanedData['STATEMENT'].map(lambda x: objCleanStatements.text_to_wordlist(x, remove_stopwords=False, stem_words=False))
    
    #call model and generate score
    objModel = modelScore()
    verityScore = objModel.execModel(dfCleanedDataSen)
    dfStatements['VERITY'] = verityScore
    dfStatements
    
    ## Prepare dataframe for table load 
    data = dfStatements[['ID','VERITY']]
    crnt_itr = int(curItr['ITR']) + 1
    data['ITR'] = crnt_itr 
    
    #insert and update the score and ratings in database
    sqlObj.insert_statement(data)
    sqlObj.update_statement(crnt_itr)
    
main()
