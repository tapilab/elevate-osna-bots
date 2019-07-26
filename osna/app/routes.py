from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path, clf_path
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack # "horizontal stack"

import pickle
import sys

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
        tweets = [t['full_text'] for t in twapi._get_tweets('screen_name', input_field, limit=200)]
        prediction = get_prediction(tweets)
        print('for user' + input_field + 'prediction = ' + prediction)
        return render_template('myform.html', title='', form=form, tweets=tweets, prediction=prediction)
    return render_template('myform.html', title='', form=form, prediction='?')

def get_prediction(tweets):
    feature_dicts = []
    texts = tweets
    tweets_texts = str(texts)
    count_mention = 0
    count_url = 0
    factor = 100
    features = {}
    for s in texts:
        if 'http' in s:
            count_url += 1
        if '@' in s:
            count_mention += 1
    features['tweets_avg_urls'] = factor * count_url / len(texts)
    features['tweets_avg_mentions'] = factor * count_mention / len(texts)
    feature_dicts.append(features)

    X_words = count_vec.transform([tweets_texts])
    X_features = dict_vec.transform(feature_dicts)
    X_all = hstack([X_features, X_words]).tocsr()
    prediction = clf.predict(X_all)[0]
    return prediction
