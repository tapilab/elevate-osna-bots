from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path, clf_path

import pickle
import sys

twapi = Twitter(credentials_path)
clf, vec = pickle.load(open(clf_path, 'rb'))
print('read clf %s' % str(clf))
print('read vec %s' % str(vec))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        print(input_field)
        tweets = [ t['full_text'] for t in twapi._get_tweets('screen_name', input_field, limit=200)]
        return render_template('myform.html', title='', form=form, value=tweets)
        #return redirect('/index')
    return render_template('myform.html', title='', form=form)
