#!pip install allennlp

import math
import pandas as pd
import numpy as np
import allennlp
import json

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

import nltk
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub

nltk.download('stopwords')
nltk.download('punkt')
archive = load_archive(
  "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
)

SESSIONS = {}


def get_model(model, settings=None):
    if model not in SESSIONS:
        path, loader = MODELS[model]
        SESSIONS[model] = loader(path)

    return SESSIONS[model]


def load_use(path):
    # tf.device('/cpu:0')     # :)
    g = tf.Graph()

    with g.as_default():
        placeholder = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(path)
        op = embed(placeholder)
        init = [tf.global_variables_initializer(), tf.tables_initializer()]
        init_op = tf.group(init)

    g.finalize()
    session = tf.Session(graph=g)
    session.run(init_op)

    return session, op, placeholder


MODELS = {
    'UniversalSentenceEncoder':
    ('https://tfhub.dev/google/universal-sentence-encoder-large/3', load_use),
}


def sentences2vec(sentences, model=None):
    if model is None:
        model = 'UniversalSentenceEncoder'

    sess, op, placeholder = get_model(model)
    vectors = sess.run(op, feed_dict={placeholder: sentences})

    return vectors


# Build tf-idf vectorizer 
def build_vectorizer(sentences, vocab=None, min_df=0.0, max_df=1.0, ngram_range=(1,1)):   # for a 2-gram use: ngram_range=(1,2)
 
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Build count vectorizer
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, vocabulary=vocab, ngram_range=(1,1),stop_words=stopwords)  # stop_words='english, max_features=N_FEATURES 
    cvec = count_vectorizer.fit(sentences)

    # Get feature names
    feature_names = cvec.get_feature_names()

    # Get bag-of-words and analyze
    bag_of_words = cvec.transform(sentences)
    
    # Transform bag_of_words into tf-idf matrix
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(bag_of_words)


    sentence_word_count= np.asarray(bag_of_words.sum(axis=1)).ravel().tolist() 
 
    return  feature_names, tfidf, sentence_word_count 
  
def eliminate_intros(sent):
    result=sent['sentence']
    comma_location=sent['sentence'].find(',')
    #if there is a comma at the beginning of the phrase
    if( comma_location<30 & comma_location>=0):
      #split the phrase into sentences 
      predictor = Predictor.from_archive(archive, 'constituency-parser')
      to_test=predictor.predict_json(sent)['trees']
      #print(json.dumps(to_test, indent=4, sort_keys=True))
      candidate=to_test[0:to_test.find('(, ,)')];
      openp=candidate.count('(')
      closedp=candidate.count(')')
      #if the comma actually splits the sentence tree in two eliminate everything up to the comma
      if(openp>closedp):
        result=result[result.find(',')+1:len(result)]
        result=result.strip()
        result=result.capitalize() 
      #print(result)
    return result
  
def summarize_text(text, target_len=None, keep=None):
    keep = keep or []
    sentences = sent_tokenize(text)
    
    feature_names, tfidf,  sentence_word_count = build_vectorizer(sentences)
    
    max_words=np.amax(sentence_word_count)

    #df_tfidf = pd.DataFrame(tfidf.todense(), columns=feature_names)

    sentence_impact=tfidf.sum(axis=1)
      
    if not target_len:
        target_len = math.ceil((math.sqrt(len(sentences)))) or 1

    assert target_len <= len(sentences)

    sent_v = sentences2vec(sentences)
    k_model = KMeans(n_clusters=target_len, random_state=0)
    k_model = k_model.fit(sent_v)
    summary = []
  
    # get a list of indexes: for each center, which is the index for the
    # closest sentence to that center
    closest, _ = pairwise_distances_argmin_min(k_model.cluster_centers_,sent_v)

    # given a list of clusters, find which sentence sits in the middle,
    # for the sentences belonging to that cluster
    
    avg = []
    for j in range(target_len):
        idx = np.where(k_model.labels_ == j)[0]
        avg.append(np.mean(idx))
        
    # lookup "table" for averages
    ordering = sorted(range(target_len), key=lambda k: avg[k])
    sum_sents = [sentences[closest[i]] for i in ordering]

    summary = {
        'preview': '\n'.join(sum_sents),
        'sentences': [],
    }

    clusters = [[]for __ in range(target_len)]
    
    distance_matrix=pairwise_distances(sent_v, k_model.cluster_centers_ )
    evaluated_sentences = []
    count=0;
    for i, sent in enumerate(sentences):
        sumimpact=sentence_impact.item(i)
        length=sentence_word_count[i]
        distance=np.amin(distance_matrix[i])
        #this rank variable contains the actual evaluation of the phrase
        rank=(sumimpact+0.5*sumimpact/length)/(3*distance+1)
        line = {
            'text': eliminate_intros({"sentence":sent}),
            #'is_summary': i in closest,
            'is_summary':True,
            'keep': i in keep,
            #'sumWeight':sumimpact,
            #'lengthcount':length,
            #'avg_dist':avg[i],
            #'distance': distance,
            'rank':rank,
            'index':i
            
        }
        if i in closest:
          count=count+1
        c = int(k_model.labels_[i])
        clusters[c].append(line)
        evaluated_sentences.append(line)

    
    #summary['sentences'] = clusters

    
    sorted_by_rank=sorted(evaluated_sentences, key=lambda k : -k['rank'])
    res = sorted(sorted_by_rank[: target_len] , key=lambda k : k['index'])
    #print('sorted by rank',len(res))
    #for i, sent in enumerate(res):
    #  print(sent['rank'])
    #  print(sent['text'])
    #  print('\n')
    #print('______________')
    
  
    return res
