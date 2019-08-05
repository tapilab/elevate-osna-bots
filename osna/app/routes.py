from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path, clf_path
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack # "horizontal stack"
from ..cli import get_tweets_features
import pickle
import sys
import numpy as np

twapi = Twitter(credentials_path)
clf, count_vec, dict_vec = pickle.load(open(clf_path, 'rb'))
print('read clf %s' % str(clf))
print('read count_vec %s' % str(count_vec))
print('read dict_vec %s' % str(dict_vec))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        print(input_field)
        tweet_objects = [t for t in twapi._get_tweets('screen_name', input_field, limit=200)]
        tweets = [t['full_text'] for t in tweet_objects]
        if len(tweet_objects)==0:
            return render_template('myform.html', title='', form=form, prediction='?', confidence='?')
        X_all, prediction = get_prediction(tweet_objects)
        print('for user' + input_field + 'prediction = ' + prediction)
        # calculate confidence
        probas = clf.predict_proba(X_all)
        print('probas=', probas)
        confidence = round(probas.max(), 2)
        print('predicted %s with probability %.2f' % (prediction, confidence))
        print_top_features(X_all)
        return render_template('myform.html', title='', form=form, tweets=tweets, prediction=prediction, confidence=confidence)
    return render_template('myform.html', title='', form=form, prediction='?', confidence='?')

def get_prediction(tweet_objects):
    tweets = [t['full_text'] for t in tweet_objects]
    user = tweet_objects[0]['user']
    followers_count = user['followers_count']
    listed_count = user['listed_count']
    friends_count = user['friends_count']
    default_profile_image = int(user['default_profile_image'])
    default_profile = int(user['default_profile'])

    feature_dicts = []
    features = get_tweets_features(tweets, tweets, len(tweet_objects), followers_count, listed_count, friends_count, default_profile_image, default_profile)
    feature_dicts.append(features)
    X_features = dict_vec.transform(feature_dicts)
    X_words = count_vec.transform([str(tweets)])
    X_all = hstack([X_features, X_words]).tocsr()
    prediction = clf.predict(X_all)[0]
    return X_all, prediction

def print_top_features(X_all):
    coef = [-clf.coef_[0], clf.coef_[0]]
    features = dict_vec.get_feature_names() + count_vec.get_feature_names()
    # why was the first example labeled bot/human?
    for i in np.argsort(coef[0][X_all[0].nonzero()[1]])[-1:-11:-1]:
        idx = X_all[0].nonzero()[1][i]
        print(features[idx])
        print(coef[0][idx])