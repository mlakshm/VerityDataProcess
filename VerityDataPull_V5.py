import os
import subprocess
import sys
os.environ['DYLD_LIBRARY_PATH'] = "/anaconda3/lib/python3.7/site-packages/clidriver/lib:$DYLD_LIBRARY_PATH"
#debussy = int(os.environ.get('DYLD_LIBRARY_PATH', 'Not Set'))
#print (debussy)
# General:
sys.stdout.flush()

import numpy as np
import pandas as pd
from collections import Counter
import ast
from itertools import chain
from fuzzywuzzy import fuzz, process
from langdetect import detect
import json
import jhashcode
import random
from datetime import datetime
import smtplib
import language_check
from langdetect import detect
from langdetect import DetectorFactory 
DetectorFactory.seed = 0
import codecs
import traceback
import collections

# Logging:

import logging

logging.basicConfig(filename='verity.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')
logging.root.setLevel(logging.INFO)
logger=logging.getLogger(__name__)
logger.info("test")
# NLTK:
import re, nltk, spacy, gensim
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Watson NLU:
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features,MetadataOptions,SentimentOptions, SemanticRolesOptions, RelationsOptions, CategoriesOptions, ConceptsOptions, KeywordsOptions, EmotionOptions, EntitiesOptions

# Watson Language Translation:
from ibm_watson import LanguageTranslatorV3

# ElasticSearch:
from elasticsearch import RequestsHttpConnection, Elasticsearch

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from textblob import TextBlob
from sklearn.manifold import TSNE

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim
from IPython.display import display
from tqdm import tqdm

#ibm_db
#export DYLD_LIBRARY_PATH=/anaconda3/lib/python3.7/site-packages/clidriver/lib:$DYLD_LIBRARY_PATH
#!install_name_tool -change /anaconda3/lib/python3.7/site-packages/libdb2.dylib /anaconda3/lib/python3.7/site-packages/clidriver/lib/libdb2.dylib /anaconda3/lib/python3.7/site-packages/ibm_db.cpython-37m-darwin.so 
import ibm_db
import ibm_db_dbi

# Visualization
import seaborn as sb
import matplotlib.pyplot as plt
#from bokeh.plotting import figure, output_file, show
#from bokeh.models import Label
#from bokeh.io import output_notebook

# Env File Loading
from dotenv import load_dotenv
#load_dotenv()

# OR, the same with increased verbosity
#load_dotenv(verbose=True)

# OR, explicitly providing path to '.env'
from pathlib import Path  # python3 only
#env_path = Path('.') / 'VerityDataPullold.env'
env_path = Path('.') / 'VerityDataPull.env'
#env_path="/Users/mlakshm/VerityDataPull.env"
load_dotenv(dotenv_path=env_path)

### Handle Stdout Encoding
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co|www)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|The\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
caps = "([A-Z])"
small ="([a-z])"
prepos ="(of)"
deps="(Department|Bureau|Board|Society)"
digits = "([0-9])"

index="IndexPosition"
#es=Elasticsearch(['https://blueminedev.w3-969.ibm.com/es'],connection_class=RequestsHttpConnection, http_auth=('elastic', 'WEmhzce6yxUoArKEOy'), use_ssl=True, verify_certs=False)
es=Elasticsearch([os.getenv("es_url")],connection_class=RequestsHttpConnection, http_auth=(os.getenv("es_uname"), os.getenv("es_pass")), use_ssl=True, verify_certs=False)
print(es)

# Given the response of Watson in JSON, following function treats raw response as dictionary and gets the "feature".
def trimmer(resp,feature): 
    result = resp[feature];
    return result
def elasticDocsPull(i,lastrundate):
# ONLY FOR EXTERNAL REPORTS! 
    try: 
        
       # res = es.search(index='bm_latest_idc_reports'.format(index), doc_type="document", _source=["title","summary", "content", "url", "publish-date","deleted","source","access-control"],body={ 
        res = es.search(index='bluemine_for_verity'.format(index), doc_type="document", _source=["title","summary", "content", "url", "publish-date","deleted","source","access-control"],body={
        "from":i, "size":'1000',
        "query": {
        "bool":{
        "must":[
          {
          "match": {
          "source.source-group": "IDC"
          }
          },
          ],
        "filter":
        [
        {
        "range": {
        "last-index-date": {
        "gte":int(lastrundate),
        "lte":"20200115000000"
        }
        }
        }
        ]
      }
     }
    }
        )
        aa = []
        ids = []
        documents = []
        for i in range(len(res['hits']['hits'])):
            b = res['hits']['hits'][i]["_source"]['source']['source-id']
            c = str(b)
            a = res['hits']['hits'][i]["_source"]['content']+"****"+res['hits']['hits'][i]['_id']+"~"+res['hits']['hits'][i]["_source"]['url']+"~"+res['hits']['hits'][i]["_source"]['source']['source-name']+"~"+res['hits']['hits'][i]["_source"]['source']['source-group']+"~"+c+"~"+res['hits']['hits'][i]["_source"]['publish-date']+"~"+res['hits']['hits'][i]["_source"]['access-control']['confidentiality']+"~"+res['hits']['hits'][i]["_source"]['deleted']+"~"+res['hits']['hits'][i]["_source"]['title']           
            aa.append(a)
        return aa
    except:
        pass


def elasticDocsPullInternal(i,lastrundate):
# ONLY FOR INTERNAL REPORTS! 
    res = es.search(index='internalreports_prod_04242002'.format(index), doc_type="document", _source=["title","summary", "content", "url", "publish-date","deleted","source","access-control"],body={     
    "from":i, "size":'1000',
    "query": {
    "bool": {
      "must": [
        {
          "match": {
            "source.source-id": ""
          }
        }
      ], 
      "must_not": [
        {
          "match": {
            "deleted": "true"
          }
        }
      ], 

      "should": [
        {
          "match": {
            
            "language":"en"
          }
        },
        {
          "match": {
            "language":""
          }
        }
        
      ],
        
      "filter":
        [
        {
        "range": {
        "last-index-date": {
        "gte":int(lastrundate),
        "lte":"now"
        }
        }
        }
        ]
    } 
    }
}
    )
    aa = []
    ids=[]
    for i in range(len(res['hits']['hits'])):
        b = res['hits']['hits'][i]["_source"]['source']['source-id']
        c = str(b)
        confidentiality="ibm-internal"
        try:
          confidentiality=res['hits']['hits'][i]["_source"]['access-control']['confidentiality']
        except Exception as e:
          print("Can't retrieve Confidentiality")
        a = res['hits']['hits'][i]["_source"]['content']+"****"+res['hits']['hits'][i]['_id']+"~"+res['hits']['hits'][i]["_source"]['url']+"~"+res['hits']['hits'][i]["_source"]['source']['source-name']+"~"+res['hits']['hits'][i]["_source"]['source']['source-group']+"~-1~"+res['hits']['hits'][i]["_source"]['publish-date']+"~"+confidentiality+"~"+res['hits']['hits'][i]["_source"]['deleted']+"~"+res['hits']['hits'][i]["_source"]['title'] 
        aa.append(a)

    return aa


def elasticDocsPullMul(i,lastrundate,srcgp):
# ONLY FOR EXTERNAL REPORTS! 
    try: 

       # res = es.search(index='bm_latest_idc_reports'.format(index), doc_type="document", _source=["title","summary", "content", "url", "publish-date","deleted","source","access-control"],body={
        res = es.search(index='bluemine_for_verity'.format(index), doc_type="document", _source=["title","summary", "content", "url", "publish-date","deleted","source","access-control"],body={
        "from":i, "size":'1000',
        "query": {
        "bool":{
        "must":[
          {
          "match": {
          "source.source-group": srcgp
          }
          },
          ],
        "must_not": [
        {
          "match": {
            "type": "Investment reports"
          }
        },
        {
          "match": {
            "access-control.confidentiality": "ibm-confidential"
          }
        },
        {
          "terms": {
            "source.source-id": [
              "1047",
              "1121"
            ]
          }
        }
      ],
        "filter":
        [
        {
        "range": {
        "last-index-date": {
        "gte":int(lastrundate),
        "lte":"now"
        }
        }
        }
        ]
      }
     }
    }
        )
        aa = []
        ids = []
        documents = []
        for i in range(len(res['hits']['hits'])):
            b = res['hits']['hits'][i]["_source"]['source']['source-id']
            c = str(b)
            a = res['hits']['hits'][i]["_source"]['content']+"****"+res['hits']['hits'][i]['_id']+"~"+res['hits']['hits'][i]["_source"]['url']+"~"+res['hits']['hits'][i]["_source"]['source']['source-name']+"~"+res['hits']['hits'][i]["_source"]['source']['source-group']+"~"+c+"~"+res['hits']['hits'][i]["_source"]['publish-date']+"~"+res['hits']['hits'][i]["_source"]['access-control']['confidentiality']+"~"+res['hits']['hits'][i]["_source"]['deleted']+"~"+res['hits']['hits'][i]["_source"]['title']
            aa.append(a)
        return aa
    except:
        exp_tb=traceback.format_exc()
        print(exp_tb)
        pass

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def genSentences(elas):
            splitraw = re.split(r'([?!.]\s*)', elas)
            splitlist =[]
            for i in range(0,len(splitraw)-2,2):
               sentence = splitraw[i]
               delim = splitraw[i+1]
               splitlist.append(sentence+delim)
            return splitlist 

def genSentencesImproved(elas):
      #logger.info("Generating Sentences...")
      sentences = []
      import re
      alphabets= "([A-Za-z])"
      prefixes = "(Mr|St|Mrs|Ms|Dr|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.]"
      suffixes = "(Inc|Ltd|Jr|Sr|Co|www)"
      starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|The\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
      acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
      websites = "[.](com|net|org|io|gov)"
      caps = "([A-Z])"
      small ="([a-z])"
      prepos ="(of)"
      deps="(Department|Bureau|Board|Society)"
      digits = "([0-9])"
      #htmlheader="(</h2>|</li>|</p>)"
      #tempstop="xstopx"

      try:
          text = " " + elas + "  "
          text = text.replace("\n"," ")
          text = re.sub(prefixes,"\\1<prd>",text)
          text = re.sub(websites,"<prd>\\1",text)
          text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
          if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
          text = re.sub("\s" + alphabets + "[.] "," \\1<prd> The ",text)
          text = re.sub(acronyms+" "+starters,"\\1 <stop> \\2",text)
          text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
          text = re.sub(alphabets + "[.]" + alphabets + "[.]"+" "+deps+" "+prepos,"\\1<prd>\\2<prd> \\3 \\4",text)
          text = re.sub(alphabets + "[.]" + alphabets + "[.]"+" "+caps,"\\1<prd>\\2<prd> <stop> \\3",text)
          #text = re.sub('(http\S+)', '\1', text)
          text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
          text = re.sub(" "+suffixes+"[.] "+starters," \\1 <stop> \\2",text)
          text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
          text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)  
          text = text.replace(". ",". <stop>")
          text = text.replace("? ","? <stop>")
          text = text.replace("! ","! <stop>")
          text = text.replace("▪"," <stop>▪")
          text = text.replace("•"," <stop>•")
          text = text.replace("<prd>",".")
          sentences = text.split(" <stop>")
          sentences = sentences[:-1]
          sentences = [s.strip() for s in sentences]
      except Exception as e:
          exp_tb=traceback.format_exc()
          print(exp_tb)
          print(e)
          subj="Error generating Sentences from the Document"
          sendmail(subj,exp_tb)
          logger.error(exp_tb)
          sys.exit(1) 
      return sentences

def execQuery(conn, schemaName, qry, msg):
                            logger.info("Executing Query...")
                            docId=""
                            rows=""
                            selectDocumentStatement = qry;
                            print("here***************************************")
                            print(qry)
                            try:
                               cur = conn.cursor()
                               cur.execute(selectDocumentStatement)
                            except Exception as e:
                               exp_tb=traceback.format_exc()
                               print(exp_tb)
                               print(e)
                               subj=msg
                               sendmail(subj,exp_tb)
                               logger.error(exp_tb)
                               sys.exit(1)
                            else:
                               qlist=qry.split(" ");
                               first_word = qlist[0].lower()
                               if ( first_word == "delete" ) or ( first_word == "update" ):
                                  print("It's a delete or an update query")
                               else:
                                   rows = cur.fetchall()
                            return rows

def insertDocs(meta, conn, schemaName):
    logger.info("Inserting Doc...")
    documentRecordId=""
    if len(meta)==9:
                 #           logger.info("Inserting...")
                            print(meta[7])
                            sIsDeleted = meta[7].lower()
                            print(sIsDeleted)
                            if(sIsDeleted=="true"):
                                  sIsDeleted=1
                            elif(sIsDeleted=="false"):
                                  sIsDeleted=0

                            print(sIsDeleted)
                            print("**********")
                            insertDocumentStatement = "INSERT INTO " + schemaName + ".DOCUMENT (NL_DOC_ID, TITLE, URL, SOURCE_NAME, SOURCE_GROUP, SOURCE_ID, PUBLISHED_DATE, CONFIDENTIALITY, DELETED_FG) VALUES('" + meta[0] + "', ?, '" + meta[1] + "', '" + meta[2] + "', '" + meta[3] + "', " + meta[4] + ", '" + meta[5] + "', '" + meta[6] + "', " + str(sIsDeleted) + ")"
                            insertDocumentSqlStatement = "SELECT ID FROM NEW TABLE ( " + insertDocumentStatement + " )"
                            print(insertDocumentSqlStatement)
                            try:
                               cur = conn.cursor()
                               cur.execute(insertDocumentSqlStatement, ( str(meta[8]), ) )
                            except Exception as e:
                               exp_tb=traceback.format_exc()
                               print(exp_tb)
                               print(e)
                               subj="Inserting Document Failed "+meta[0]
                               #sendmail(subj,exp_tb)
                               logger.error(exp_tb)
                               sys.exit(1)
                            else:
                               row = cur.fetchall()
                               documentRecordId = row[0][0]
                               print("***doc_rec_id***")
                               print(documentRecordId)
    return documentRecordId   


def insertMultiStatement(conn, multi_qry, multi_qry1):
                                logger.info("Insert Multi Value Statement...")
                                try:
                                    cur = conn.cursor()
                                   # cur.execute(multi_qry)
                                    cur.execute(multi_qry, (multi_qry1))
                                    return 0
                                    #cur.execute(insertStatementSqlStatement, (item, statementHashcode, documentRecordId, rating, numberOfRatings, sIsDeleted, item, statementHashcode, documentRecordId, rating, numberOfRatings, sIsDeleted))
                                except ibm_db_dbi.IntegrityError:
                                    exp_tb=traceback.format_exc()
                                    print(exp_tb)
                                    print('ibm_db_dbi.IntegrityError', ibm_db.stmt_errormsg())
                                    #print(item)
                                    print("Skipping....")
                                    return 1
                                except Exception as e:
                                    print("*********************here*****************************")
                                    exp_tb=traceback.format_exc()
                                    print(exp_tb)
                                    subj="Inserting Statements for a Document Failed"
                                    statement_info=" Select Insert Statement: "+multi_qry
                                    logger.info(statement_info)
                                    logger.error(exp_tb)
                                    return 1


def insertStatement(meta, conn, schemaName, item):
                                logger.info("Insert Statement...")

                                sIsDeleted = meta[7].lower()
                                if(sIsDeleted=="true"):
                                    sIsDeleted=1
                                elif(sIsDeleted=="false"):
                                    sIsDeleted=0
                                statementHashcode = jhashcode.hashcode(item);
                                rating = 0
                                numberOfRatings = 0
                                ratingstr = str(rating)
                                numratingstr = str(numberOfRatings)
                                insertStatementStatement = "INSERT INTO " + schemaName + ".STATEMENT (STATEMENT, STATEMENT_HASHCODE, DOCUMENT_ID, STATEMENT_TYPE_ID, RATING, NUMBER_OF_RATINGS, FACT_FG, USER_SUBMITTED_FG, MDI_PICK_FG, DELETED_FG) VALUES(?, ?, ?, 0, ? , ? , 0, 0, 0, ?)"
                                insertStatementSqlStatement = "SELECT ID FROM NEW TABLE ( " + insertStatementStatement + " )"
                                
                                #print(insertStatementSqlStatement)

                                try:
                                    cur = conn.cursor()
                                    cur.execute(insertStatementSqlStatement, (item, statementHashcode, documentRecordId, rating, numberOfRatings, sIsDeleted))
                                    #cur.execute(insertStatementSqlStatement, (item, statementHashcode, documentRecordId, rating, numberOfRatings, sIsDeleted, item, statementHashcode, documentRecordId, rating, numberOfRatings, sIsDeleted))
                                except ibm_db_dbi.IntegrityError:
                                    exp_tb=traceback.format_exc()
                                    print(exp_tb)
                                    print('ibm_db_dbi.IntegrityError', ibm_db.stmt_errormsg())
                                    print(item)
                                    print("Skipping....")
                                    #sys.exit(1)
                                except Exception as e:
                                    print("*********************here*****************************")
                                    exp_tb=traceback.format_exc()
                                    print(exp_tb)
                                    print(e)
                                    subj="Inserting Statement Failed"
                                    statement_info=" Statement: "+item+" Select Insert Statement: "+insertStatementSqlStatement
                                    logger.info(statement_info)
                                    logger.error(e)
                                   # sys.exit(1)
                                else:
                                    row = cur.fetchall()
                                    statementRecordId = row[0][0]
     #                               print(statementRecordId)

def getLastRunDate(pathlastrun):
    logger.info("Obtaining the last run date...")
    lastrundate=""
    try:
      with open(pathlastrun) as infile:
         for line in infile:
             lastrundate=line
    except Exception as e:
      exp_tb=traceback.format_exc()
      print(exp_tb)
      print(e)
      sendmail("Error Retrieving Last Run Date",exp_tb)
      logger.error(e)
      sys.exit(1)
    print(lastrundate)
    logger.info(lastrundate)
    return lastrundate
                
    
def pullDocs(lastrundate):
    logger.info("Pulling from Elastic Search...")
    for i in [j for j in range(0, 9)]:
       print("ival")
       if i==0:
           x=0
       else :
           print("here")
           x=x+1000
       print(x)
       bodiesArray = elasticDocsPull(x,lastrundate)
       print("Out of 1000 DocIDs queried, elastic gives us ", len(bodiesArray), " reports back.")
       fromElastic.append(len(bodiesArray))

       metaArr.append(bodiesArray)

    sumfromElastic = sum(fromElastic)
    print("Total Docs Queried:", 10000) 
    print("Total Received from Elastic:", sumfromElastic)
    #print("Total Loss from Elastic:", 10000 - sumfromElastic)
    return sumfromElastic

def pullDocsMultipleSrc(lastrundate,srclist):
    logger.info("Looping to pull the docs....")
    for srcgpraw in srclist:
       srcgp=srcgpraw.strip()
       print("*****")
       print(srcgp)
       print("*****")
       for i in [j for j in range(0, 9)]:
         print("ival")
         if i==0:
           x=0
         else :
           logger.info("Iterating - inside the loop....")
           x=x+1000


         print(x)
         bodiesArray=[]
         if srcgp=="Internal":
            bodiesArray = elasticDocsPullInternal(x,lastrundate)
         else:
            bodiesArray = elasticDocsPullMul(x,lastrundate,srcgp)
         print("Out of 1000 DocIDs queried, elastic gives us ", len(bodiesArray), " reports back.")
         fromElastic.append(len(bodiesArray))

         metaArr.append(bodiesArray)

       sumfromElastic = sum(fromElastic)
       print("Total Docs Queried:", 10000) 
       print("Total Received from Elastic:", sumfromElastic)
       #print("Total Loss from Elastic:", 900 - sumfromElastic)
       #return sumfromElastic

def connectToDB():
    logger.info("Connecting to the Database....")
    conn=""
    try:
#       db2_conn = ibm_db.connect("HOSTNAME=db2w-jofygxw.us-south.db2w.cloud.ibm.com;PORT=50001;PROTOCOL=TCPIP;DATABASE=BLUDB;UID=vapp;PWD=password4Verity@app;SECURITY=SSL", "", "")
       db2_conn = ibm_db.connect(os.getenv("db2_URL"), "", "")
       print(db2_conn)
       conn = ibm_db_dbi.Connection(db2_conn)
    except Exception as e:
       exp_tb=traceback.format_exc()
       print(exp_tb)
       sendmail("Error Retrieving Last Run Date",exp_tb)
       logger.error(e)
       sys.exit(1) 
    return conn

def sendmail(sub, body):
    try:
       logger.info("Sending out the email....")
       SERVER = "us.relay.ibm.com"
       FROM = "verity@bluemine.ibm.com"
       TO = ["mlakshm@us.ibm.com"] # must be a list
       SUBJECT = sub
       TEXT = body
       message = 'From:'+FROM+\
       '\nTo: '+", ".join(TO)+\
       '\nSubject: '+SUBJECT+'\n'+TEXT
       print(message)
       server = smtplib.SMTP(SERVER)
       server.sendmail(FROM, TO, message)
       server.quit()
       return 0
    except Exception as e:
       exp_tb=traceback.format_exc()
       print(exp_tb)
       print(exp_tb)
       logger.error(exp_tb)

def sendmailold(sub, body):
    try:
       logger.info("Sending out the email....")
       SERVER = "localhost"
       FROM = "verity@bluemine.ibm.com"
       TO = ["mlakshm@us.ibm.com"] # must be a list

       SUBJECT = sub

       TEXT = body

       # Prepare actual message

       message = """\
       From: %s
       To: %s
       Subject: %s

       %s
       """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
       server = smtplib.SMTP(SERVER)
       server.sendmail(FROM, TO, message)
       server.quit()
       return 0 
    except Exception as e:
       exp_tb=traceback.format_exc()
       print(exp_tb)
       #print(e)
       logger.error(e)   


def existingDocs(schemaName, tableName):

    doc_id_list=[]
    ref_id_list=[]
    doc_id_set={}
    doc_ref_dict=dict.fromkeys([''])

    try:
        logger.info("Pulling the Existing Doc Info from the Database....")
        qry_pull_docs="select ID, NL_DOC_ID from " + schemaName + "." + tableName;
        errmsg="Pulling Existing Document IDs Failed "
        existDocs=execQuery(conn, schemaName, qry_pull_docs, errmsg)
    
        logger.info("Loading the Existing Doc info to lists, set and Dictionary....")
       
        for doc in existDocs:
           ref_id_list.append(doc[0])
           doc_id_list.append(doc[1])
           doc_ref_dict.update([ (doc[1], doc[0]) ] )
           doc_id_set=set(doc_id_list)
    except Exception as e:
        exp_tb=traceback.format_exc()
        print(exp_tb)
        sendmail("Error Pulling/Processing Existing Doc Info from the Database",exp_tb)
        logger.error(e)
        sys.exit(1) 
            
    return doc_id_list, ref_id_list, doc_id_set, doc_ref_dict

def populateMultiStatements(exist_check,splitlist,sentence_list,metadata,documentRecordId):
 logger.info("Populating Statements in Bulk...")
 meta = metadata.split("~")
 if len(meta)==9:
    sIsDeleted = meta[7].lower()
    if(sIsDeleted=="true"):
         sIsDeleted=1
    elif(sIsDeleted=="false"):
         sIsDeleted=0
    #statementHashcode = jhashcode.hashcode(item);
    statementtypeid = 0
    statementtypeid_str = str(statementtypeid)
    rating = 0
    numberOfRatings = 0
    ratingstr = str(rating)
    numratingstr = str(numberOfRatings)
    fact_flag = 0
    fact_flag_str = str(fact_flag)
    user_submitted_flag = 0
    user_submitted_flag_str = str(user_submitted_flag)
    mdi_pick_fg = 0
    mdi_pick_fg_str = str(mdi_pick_fg)
    multi_qry =  "INSERT INTO " + schemaName + ".STATEMENT (STATEMENT, STATEMENT_HASHCODE, DOCUMENT_ID, STATEMENT_TYPE_ID, RATING, NUMBER_OF_RATINGS, FACT_FG, USER_SUBMITTED_FG, MDI_PICK_FG, DELETED_FG) \n"
    vals = str(documentRecordId) + "," +statementtypeid_str + "," +ratingstr +","+ numratingstr +","+ fact_flag_str +","+ user_submitted_flag_str +","+ mdi_pick_fg_str +","+ str(sIsDeleted)
    try:
       #print(splitlist)
       counter=0;
       if( (len(splitlist) > 0) and (sIsDeleted==0)):
         qry_arr = []
         qry_arr.append(multi_qry)
         qry_arr1 = []
         detect_new=0
         new_sentences = [] 
         for item in splitlist:
                           statementHashcode = jhashcode.hashcode(item);
                           qry_part=""
                           qry_part1=""
                           if exist_check==0: ### Inserting sentences from new doc
                               #logger.info("here")
                               detect_new = 1
                               new_sentences.append(item)
                               if( counter == 0): 
                                    qry_part =  "VALUES( ?," +  str(statementHashcode) + "," +vals + ")\n"
                                    qry_part1 = item + ", "
                               elif ( counter < len(splitlist)-1  ):
                                    qry_part = "UNION ALL\n VALUES( ?," + str(statementHashcode) + "," +  vals + ")\n"
                                    qry_part1 = item + ", "
                               else :
                                    qry_part = "UNION ALL\n VALUES( ?," + str(statementHashcode) + "," +  vals + ")"
                                    qry_part1 = item               
                               counter=counter+1;
                           elif exist_check==1: ### Inserting new sentences from updated doc
                               if item not in sentence_list:
                                  detect_new = 1
                                  new_sentences.append(item)
                                  #print(item)
                                  if( counter == 0):
                                      qry_part = "VALUES( ?," + str(statementHashcode)  + "," +  vals + ")\n"
                                      qry_part1 = item + ", "  
                                  elif ( counter < len(splitlist)-1  ):
                                      qry_part = "UNION ALL\n VALUES( ?," + str(statementHashcode) +  "," + vals + ")\n"
                                      qry_part1 = item + ", " 
                                  else :
                                      qry_part = "UNION ALL\n VALUES( ?," + str(statementHashcode) +  "," +vals + ")"
                                      qry_part1 = item  
                                  counter=counter+1;
                          
                           qry_arr.append(qry_part)
                           multi_qry = ''.join(qry_arr)

         #print(multi_qry)
         if detect_new==1:
            ret_val=insertMultiStatement(conn,multi_qry,new_sentences)
            if ( ret_val == 1 ):
                #print("Inserting each statement....") 
                populateStatements(exist_check,splitlist,sentence_list,metadata,documentRecordId)
                
                
        
    except Exception as e:
         exp_tb=traceback.format_exc()
         print(exp_tb)
         sendmail("Error Retrieving Populating Statements to the Database",exp_tb)
         print(e)
         logger.error(e)
         sys.exit(1)

def populateStatements(exist_check,splitlist,sentence_list,metadata,documentRecordId):
    logger.info("Populating Statements...")
    try:
         #print(splitlist)
         for item in splitlist:
                           if exist_check==0: ### Inserting sentences from new doc
                               sentwithmeta = item+"~"+metadata
                               meta = metadata.split("~")
                               statementRecordId=insertStatement(meta, conn, schemaName, item)
                               
                           elif exist_check==1: ### Inserting new sentences from updated doc
                               itemhash=jhashcode.hashcode(item)
                               if item not in sentence_list:
                                  #print(item)
                                  sentwithmeta = item+"~"+metadata
                                  meta = metadata.split("~")
                                  statementRecordId=insertStatement(meta, conn, schemaName, item)
    except Exception as e:
         exp_tb=traceback.format_exc()
         print(exp_tb)
         sendmail("Error Retrieving Populating Statements to the Database",exp_tb)
         logger.error(e)
         sys.exit(1) 

def processDocument(i):
     logger.info("Starting to process the document...")

     splitlist=[]
     sentence_list=[]
     exist_check=0

     try:
          #if ( (i.find(".xls") == -1) and (i.find(".ppt") == -1) ): ## Excluding xls and ppt docs 
            metasplit = i.split("****")
            elas = metasplit[0]
            metadata = metasplit[1]
            splitlist = genSentencesImproved(elas)
            meta = metadata.split("~")
            cleanedlist = []
            #stmt_ref_dict=collections.defaultdict(list)
            for item in splitlist:
             if len(meta)==9:
                try:
                   #flag1 = detect(meta[8])
                   #flag1 = isEnglish(meta[8])
                   #print(item)
                   #item_clean=''.join(e for e in item if e.isalnum())
                   flag1 = isEnglish(item)
                   flag2 = detect(item)
                   if (( flag2 != 'en') and ( flag1 == False ) ): #Excluding Other Language Sentences
                      olangarr.append(meta[1])
                      #print("OtherLang")
                      #print(item)
                   else:
                      newstr = item.replace("/ ‚àö√ß¬¨¬®‚àö√©¬¨√º",'')
                      newstr = newstr.replace("¬©2019",'')
                      newstr = newstr.replace("¬©",'')
                      newstr = newstr.replace("©2019",'')
                      newstr = newstr.replace("©2018",'')
                      newstr = newstr.replace("©2017",'')
                      newstr = newstr.replace("©2016",'')
                      newstr = newstr.replace("‚àö√ß¬¨¬®‚àö√©¬¨√º",'')
                      newstr = newstr.replace("‚àö¬¢¬¨√§¬¨√∂‚àö√©¬¨√±‚àö√©¬¨√¶",'')
                      newstr = newstr.replace("‚àö¬¢¬¨√§¬¨√∂‚àö√©¬¨√±‚àö√©¬¨¬®",'')
                      newstr = newstr.replace("‚àö¬¢¬¨√§¬¨√∂‚àö√©¬¨√±‚àö√ß¬¨‚àÇ",'')
                      newstr = newstr.replace("‚àö¬¢¬¨√§¬¨√∂‚àö√©¬¨√°‚àö√ß¬¨¬Æ",'')
                      newstr = newstr.replace("‚àö¬¢¬¨√§¬¨√∂‚àö√©¬¨√±‚àö√©¬¨‚â†",'')
                      newstr = newstr.replace("‚àö√ß¬¨¬®‚àö√ß¬¨¬©2019",'')
                      newstr = newstr.replace("‚àö√ß¬¨¬®‚àö√ß¬¨¬©",'')
                      newstr = newstr.replace("© ",'')
                      #newstr = newstr.replace("re.sub(r'▪','', txt)",'')
                      #newstr = newstr.replace('▪','')
                      #newstr = newstr.replace('•','')
                      item = newstr.replace("©",'')
                      item = re.sub(r'\b(\w+)( \1\b)+', r'\1', item)
                      testlist = item.split(" ")
                      if len(testlist)>4:
                         if (len(item)<1000):
                             #Cleaning the data
                             if ( ( item.find("Not applicable/do not use") == -1 ) and ( item.find("Not applicableldo not use") == -1 ) and (item.find("ANALYZE THE FUTURE !") == -1) and (item.find("ERROR:#REF!") == -1) and (item.find("ERROR:#DIV/0!") == -1) and (item.find(".com") == -1) and (item.find("@") == -1) and (item.find("Source:") == -1) and (item.find("See the Forrester report") == -1) and (item.find("Copyright") == -1) and (item.find("Report |") == -1) and (item.find("THIS COMMENTARY IS PUBLISHED BY FITCH SOLUTIONS MACRO RESEARCH") == -1) and (item.find("Fitch Solutions Forecast") == -1) and (item.find("Rural pop.") == -1) and (item.find("Fitch Solutions estimate") == -1) and (item.find("?") == -1) and (item.find("\"") == -1) and (item.find("Related Research:") == -1) and (item.find("Benefit Rating:") == -1)):
                             #if ((item.lower().find("gartner") > -1) or (item.lower().find("idc") > -1) or (item.lower().find("tbri") > -1) or (item.lower().find("PAC") > -1) or (item.lower().find("fitch") > -1) or (item.lower().find("forrester") > -1) or (item.lower().find("jpmorgan") > -1) or (item.lower().find("hfs research") > -1)  or (item.lower().find("insight partners") > -1) or (item.lower().find("acquisdata") > -1) or (item.lower().find("macquarie") > -1) or (item.lower().find("wright reports") > -1) or  (item.find("UBS") > -1)):
                                 #stmt_ref_dict[item].append("1")  
                                 cleanedlist.append(item)
                             
                except Exception as e:
                   #print("Language Detect Exception")
                   #print(item)
                   print(e)
            #newcleanedlist=[]
            #for k in stmt_ref_dict.keys():

              #if (len(stmt_ref_dict[k])<3):
                   #newcleanedlist.append(k)
              #else:
                   #print("potential junk repeat:")
                   #print(k)

            #splitset = set(newcleanedlist)
            splitset = set(cleanedlist)
            splitlist = list(splitset)
            hashlist = []
            for h in splitlist:
                hashlist.append(jhashcode.hashcode(h))
            print(len(splitlist))
            exist_check=0
            documentRecordId=""
            sIsDeleted=0
            if (len(meta) == 9):
              sIsDeleted = meta[7].lower()
              if(sIsDeleted=="true"):
                  sIsDeleted=1
              elif(sIsDeleted=="false"):
                  sIsDeleted=0
                  
            if meta[0] in doc_id_set:
              print("****************Existing Document that has been indexed is retrieved*******************************")
              ref_id = str(doc_ref_dict[meta[0]])
              documentRecordId=ref_id
              if ( (i.find(".xls") > -1) or (i.find(".ppt") > -1) or (sIsDeleted==1)): ## Excluding xls and ppt docs
               qry_update="UPDATE "+schemaName+".STATEMENT SET DELETED_FG='1' WHERE DOCUMENT_ID='"+ref_id+"'"
               print("Setting ppt, xls statements and deleted doc statemnets to deleted.")
               errmsg=" Updating Deleted Flag for Statements from .xls and .ppt Failed "
               execQuery(conn, schemaName, qry_update, errmsg) 
               qry_doc_update="UPDATE "+schemaName+".DOCUMENT SET DELETED_FG='1' WHERE ID='"+ref_id+"'"
               print("Setting ppt, xls and deleted docs to deleted.")
               errmsg=" Updating Deleted Flag for Statements from .xls and .ppt Failed "
               execQuery(conn, schemaName, qry_doc_update, errmsg) 
              else:               
               qry_pull_sentns = "select STATEMENT from " + schemaName + ".STATEMENT WHERE DELETED_FG='0' and DOCUMENT_ID='"+ref_id+"'"
               errmsg="Pulling Statements from Existing Document ID Failed "
               sentences=execQuery(conn, schemaName, qry_pull_sentns, errmsg)
               sentence_list=[]
               for sentence in sentences:
                   #print(sentence)
                   sentence_list.append(sentence[0])
               exist_check=1
               sentence_set=set(sentence_list)
               for sentence in sentence_list: ###Handling deletes in the updated document
                   sentencehash =  jhashcode.hashcode(sentence)
                   if not sentencehash in hashlist:
                         print("detected deleted sentence")
                         print(sentence)
                         #print(splitlist)
                         #print(sentence_list)
                         st_hash= jhashcode.hashcode(sentence)
                         #qry_pull_sentns="delete from " + schemaName + ".STATEMENT WHERE DOCUMENT_ID='"+ref_id+"' AND STATEMENT ='"+str(sentence)+"'"
                         qry_pull_sentns="UPDATE "+ schemaName +".STATEMENT SET DELETED_FG='1' WHERE STATEMENT_HASHCODE ='"+str(st_hash)+"' and DOCUMENT_ID='"+ref_id+"'"
                         errmsg="Deleting Removed Statements for Existing Document ID Failed "
                         execQuery(conn, schemaName, qry_pull_sentns, errmsg)
            else:
             if ( (i.find(".xls") == -1) and (i.find(".ppt") == -1) and (sIsDeleted==0) ): ## Excluding xls , ppt docs and deleted docs
              print("New Document is retrieved")
              documentRecordId=insertDocs(meta, conn, schemaName)
             else:
                if (len(meta)==9):
                   print("Skipped Document: "+meta[8])
                else:
                   print("Skipped Document ")
     except Exception as e:
            exp_tb=traceback.format_exc()
            print(exp_tb)
            sendmail("Error Processing the Docs Retrieved from Elasic Search",exp_tb)
            logger.error("Error Processing the Docs Retrieved from Elasic Search")
            logger.error(e)
            sys.exit(1) 

     return splitlist, sentence_list, exist_check, metadata, documentRecordId

def writeToFile(last_run_file):
    logger.info("Updating last run date...")

    with open(last_run_file, 'w') as f:
        now = str(datetime.today().strftime('%Y%m%d%H%M%S'))
        print(now)
        f.write(now)
    #sendmail("Successfully populated the data!!!","Successfully populated the data!!!")
########### MAIN #############
logger.info("Getting Started.....")
now = str(datetime.today().strftime('%Y%m%d%H%M%S'))
logger.info(now)
fromElastic = []
megaArr = []
metaArr = []

try:
    pathlastrun=os.getenv("last_run_file")
    lastrundate=getLastRunDate(pathlastrun)  ### Obtaining Last Run Date
    #srclist=["IDC","Gartner","Forrester","PAC","Fitch Solutions (prev. BMI Research)","HfS Research","TBRI","The Insight Partners","JPMorgan","Acquisdata","Macquarie Research","Wright Reports","UBS Equities"]
    logger.info("Getting the list of sources to pull docs from.....")
    srclist = []
    fname=os.getenv("src_file")
    with open(fname) as infile:  ### Obtaining Source List
        for line in infile:
            srclist.append(line)

    pullDocsMultipleSrc(lastrundate,srclist)
    conn = connectToDB()

    olangarr = []
    schemaName = os.getenv("schemaName")

    doc_id_list, ref_id_list, doc_id_set, doc_ref_dict = existingDocs(schemaName, "DOCUMENT")

    majorArr = []
    for j in metaArr:
        bodiesArray = j
        for i in bodiesArray:  ### Looping to process each document
          #if ( (i.find(".xls") == -1) and (i.find(".ppt") == -1) ): ## Excluding xls and ppt docs  
            splitlist,sentence_list,exist_check,metadata,documentRecordId=processDocument(i)
            if ( (i.find(".xls") == -1) and (i.find(".ppt") == -1) ): ## Excluding xls and ppt docs
              populateMultiStatements(exist_check,splitlist,sentence_list,metadata,documentRecordId)
          #else:
              #print("Skipping xls and ppt docs")
    conn.commit()
    conn.close()
    writeToFile(os.getenv("last_run_file"))
    print("Successfully Loaded data into the tables.....")
except Exception as e:
    exp_tb=traceback.format_exc()
    print(exp_tb)
    sendmail("Error with Verity Statement Population Process",exp_tb)
    logger.error("Error with Verity Statement Population Process")
    logger.error(e)
    sys.exit(1)
