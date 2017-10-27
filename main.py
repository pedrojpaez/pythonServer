import sre
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


vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3),max_df=0.95, min_df = 0.1, stop_words = stop_words)
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


km = KMeans(n_clusters=9, init='k-means++', max_iter=100, n_init=1,verbose=1)

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
for i in range(9):
    centroids=[]
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        centroids.append(terms[ind])
        print(' %s' % terms[ind])
        print()
    cluster_dict[i]=centroids
centroids_dict["clusters"]=cluster_dict


from sklearn.decomposition import NMF

n_samples = 2000
n_features = 1000
n_topics = 9
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    topics_dict={}
    for topic_idx, topic in enumerate(model.components_):
        topics=[]
        print("Topic #%d:" % topic_idx)
        topics =[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(" ".join(topics))
        topics_dict[topic_idx]=topics
    print()
    return topics_dict




# Fit the NMF model
print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))

nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(X)


print("\nTopics in NMF model:")
tfidf_feature_names = vectorizer.get_feature_names()
topics_dict=print_top_words(nmf, tfidf_feature_names, n_top_words)
centroids_dict["topics"]=topics_dict


import json
res=json.dumps(centroids_dict, ensure_ascii=False)

import logging

from flask import Flask, Response
import os

app = Flask(__name__)
port = int(os.getenv('PORT', 8000))

@app.route('/')
def hello():
    resp = Response(res)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    """Return a friendly HTTP greeting."""
    return resp


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500



app.run(host='0.0.0.0', port=port, debug=True)
