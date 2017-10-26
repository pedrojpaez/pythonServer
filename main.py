from __future__ import print_function

import sre, urllib2, sys, BaseHTTPServer
import requests


def parseAddress(input):
        if input[:7] != "http://":
                if input.find("://") != -1:
                        print( "Error: Cannot retrive URL, address must be HTTP")
                        sys.exit(1)
                else:
                        input = "http://" + input

        return input

def retrieveWebPage(address):
        web_handle = requests.get(address)
        return web_handle


match_set = set()

address = parseAddress('http://www.nytimes.com/pages/todayspaper/index.html')
website_handle = retrieveWebPage(address)
website_text = website_handle.text

dir = website_handle.url.rsplit('/',1)[0]
if (dir == "http:/"):
        dir = website_handle.url

matches = sre.findall('<a .*href="(.*?)"', website_text)

for match in matches:
    if 'ref=todayspaper' in match:
        if match[:7] != "http://":
                if match[0] == "/":
                        slash = ""
                else:
                        slash = "/"
                #match_set.add(dir + slash + match)
                match_set.add(match)
        else:
                match_set.add(match)

match_set = list(match_set)
match_set.sort()

from bs4 import BeautifulSoup
corpus=match_set

corpus[1]=corpus[1].replace('https','http')
website_text=[]
address = parseAddress(corpus[1])
website_handle = retrieveWebPage(address)
website_text.append(website_handle.text)

website_text=[]
for i in corpus:
    i=i.replace('https','http')
    address = parseAddress(i)
    website_handle = retrieveWebPage(address)
    website_text.append(website_handle.text)
    
    
soup=[]
for i in website_text:
    soup.append(BeautifulSoup(i, 'html.parser'))
    
    
s=" "

corpus_text=[]
for j in xrange(len(corpus)):
    seq=[]
    for i in soup[j].find_all('p', class_="story-body-text story-content"):
        seq.append(i.text)
    text= s.join(seq)
    corpus_text.append(text)
    
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus_text)
X.toarray()

analyze = vectorizer.build_analyzer()


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform(corpus_text)
feature_names = tf.get_feature_names() 


dense = tfidf_matrix.todense()
phrase_scores=[]
for i in xrange(len(corpus)):
    dense_document=dense[i].tolist()[0]
    phrase_scores.append([pair for pair in zip(range(0, len(dense_document)), dense_document) if pair[1] > 0])
    
from stop_words import get_stop_words

stop_words = get_stop_words('english')
stop_words.append("u'ms'")
stop_words.append("u'mr'")
stop_words.append("ms")
stop_words.append("mr")
stop_words.append("will")
stop_words.append("said")
stop_words.append("u'will")
stop_words.append("u'said")
stop_words.append("will")


vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0.1, stop_words = stop_words)
X = vectorizer.fit_transform(corpus_text)

from sklearn.cluster import KMeans
from sklearn import metrics
for i in xrange(15):
    km = KMeans(n_clusters=i+2, init='k-means++', max_iter=100, n_init=1,verbose=0)
    km.fit(X)
    print(i+2, "Clusters:")
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))


from time import time


km = KMeans(n_clusters=6, init='k-means++', max_iter=100, n_init=1,verbose=1)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))



print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
centroids_dict={}

cluster_dict={}
for i in range(6):
    centroids=[]
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        centroids.append(terms[ind])
        print(' %s' % terms[ind], end='')
        print()
    cluster_dict[i]=centroids
centroids_dict["clusters"]=cluster_dict


import json
res=json.dumps(centroids_dict, ensure_ascii=False)

from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps



app = Flask(__name__)
api = Api(app)

class Centroids(Resource):
    def get(self):
        return res

    
api.add_resource(Centroids, '/cents')
app.run()