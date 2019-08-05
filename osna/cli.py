# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import pickle
import sys
import gzip
import json
from collections import Counter
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import scale, StandardScaler
from scipy.sparse import hstack  # "horizontal stack"
from . import credentials_path, clf_path


@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)


def tweet_tokenizer(s):
    s = re.sub(r'#(\S+)', r'HASHTAG_\1', s)
    s = re.sub(r'@(\S+)', r'MENTION_\1', s)
    s = re.sub(r'http\S+', 'THIS_IS_A_URL', s)
    return re.sub('\W+', ' ', s.lower()).split()


def do_calculation(dir):
    bots = []
    humans = []
    folder = ['/bots', '/humans']
    for f in folder:
        paths = glob.glob(dir + f + '/*.json.gz')
        for p in paths:
            with gzip.open(p, 'r') as file:
                for line in file:
                    if f == folder[0]:
                        bots.append(json.loads(line))
                    elif f == folder[1]:
                        humans.append(json.loads(line))

    total_users = bots + humans
    df_total_users = pd.DataFrame(total_users)
    df_bots_users = pd.DataFrame(bots)
    df_humans_users = pd.DataFrame(humans)

    print("---Caculate number of unique users based on their id_str---")
    print("Number of unique users in total: ", df_total_users['id_str'].nunique())
    print("Number of unique users in bots: ", df_bots_users['id_str'].nunique())
    print("Number of unique users in humans: ", df_humans_users['id_str'].nunique())
    print("---")

    # Calculate number of unique messages
    bots_text = []
    bots_message_id = []
    for bot in bots:
        if 'tweets' in bot:
            for tweet in bot['tweets']:
                bots_message_id.append(tweet['id_str'])
                bots_text.append(tweet['full_text'])
    print("Number of unique messages in bots: ", len(set(bots_message_id)))

    humans_text = []
    humans_message_id = []
    for human in humans:
        if 'tweets' in human:
            for tweet in human['tweets']:
                humans_message_id.append(tweet['id_str'])
                humans_text.append(tweet['full_text'])
    print("Number of unique messages in humans: ", len(set(humans_message_id)))

    total_ids = bots_message_id + humans_message_id
    print("Number of unique messages in total: ", len(set(total_ids)))
    print("---")

    # Calculate number of unique words
    bots_words = []
    for wb in bots_text:
        bots_words = bots_words + tweet_tokenizer(wb)
        # for test
        if len(bots_words) > 1000:
            break

    humans_words = []
    for hw in humans_text:
        humans_words = humans_words + tweet_tokenizer(hw)
        # for test
        if len(humans_words) > 1000:
            break

    # Calculate most common words
    total_words = bots_words + humans_words
    print("Number of unique words in total: ", len(set(total_words)))
    print("50 most common words in total", Counter(total_words).most_common(50))
    print("50 most common words for bots", Counter(bots_words).most_common(50))
    print("50 most common words for humans", Counter(humans_words).most_common(50))


@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
    Read all files in this directory and its subdirectories and print statistics.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    do_calculation(directory)


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it.
    """
    print('reading from %s' % directory)
    # (1) Read the data...
    #
    df = read_data(directory)

    print("start making features...")
    # (2) Create classifier and vectorizer.
    X, dict_vec = make_features(df)
    print(dict_vec.get_feature_names())
    print("finished making features.")

    min_df = 0.01
    max_df = 0.5
    print("min_df=%.2f, max_df=%.2f"%(min_df, max_df))
    count_vec = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(3, 3))

    X_words = count_vec.fit_transform(df.tweets_texts)
    print(X_words.shape)
    optimal_X_all = hstack([X, X_words]).tocsr()
    ###scaler = StandardScaler(with_mean=False)  # optionally with_mean=False to save memory (keep matrix sparse)
    ###optimal_X_all = scaler.fit_transform(optimal_X_all)
    #optimal_X_all = scaler.fit_transform(optimal_X_all.todense())

    print("finished optimal_X_all.")

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000, C=1, penalty='l2')
    y = np.array(df.label)
    ## no reason to .fit here since you do it after cross validation. -awc
    # clf.fit(optimal_X_all, y)
    # print("finished clf fit.")

    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    truths = []
    predictions = []
    for train, test in kf.split(optimal_X_all):
        clf.fit(optimal_X_all[train], y[train])
        pred = clf.predict(optimal_X_all[test])
        accuracies.append(accuracy_score(y[test], pred))
        truths.extend(y[test])
        predictions.extend(pred)
    print('accuracy over all cross-validation folds: %s' % str(accuracies))
    print('mean=%.2f std=%.2f' % (np.mean(accuracies), np.std(accuracies)))
    print("classification_report: \n", classification_report(truths, predictions))

    # (4) Finally, train on ALL data one final time and
    # train...
    # save the classifier
    clf.fit(optimal_X_all, y)
    print_top_features(dict_vec, count_vec, clf)
    pickle.dump((clf, count_vec, dict_vec), open(clf_path, 'wb'))


def read_data(directory):
    bots = []
    humans = []
    folder = ['/bots', '/humans']
    name = '/*.json.gz'
    for f in folder:
        paths = glob.glob(directory + f + name)
        for p in paths:
            with gzip.open(p, 'r') as file:
                for line in file:
                    if f == folder[0]:
                        js = json.loads(line)
                        if 'tweets' in js:
                            bots.append(js)
                    elif f == folder[1]:
                        js = json.loads(line)
                        if 'tweets' in js:
                            humans.append(js)
    df_bots = pd.DataFrame(bots)[['screen_name', 'tweets', 'listed_count',
                                  'followers_count', 'friends_count', 'default_profile_image', 'default_profile']]
    df_bots['label'] = 'bot'
    df_humans = pd.DataFrame(humans)[['screen_name', 'tweets', 'listed_count',
                                      'followers_count', 'friends_count', 'default_profile_image', 'default_profile']]
    df_humans['label'] = 'human'
    frames = [df_bots, df_humans]
    df = pd.concat(frames)
    users = bots + humans
    tweets_texts = []
    num_of_tweets = []
    for u in users:
        tweets = u['tweets']  # a list of dicts
        num_of_tweets.append(len(tweets))
        texts = [t['full_text'] for t in tweets]
        tweets_texts.append(str(texts).strip('[]'))
    df['tweets_texts'] = tweets_texts
    df['num_tweets'] = num_of_tweets
    return df


def make_features(df):
    ## Add your code to create features.
    vec = DictVectorizer()
    feature_dicts = []
    for i, row in df.iterrows():
        tweets = row['tweets']
        texts = [t['full_text'] for t in tweets]
        features = get_tweets_features(texts, [row.tweets_texts], row.num_tweets, row.followers_count, row.listed_count, row.friends_count,
                                       row.default_profile_image, row.default_profile)
        feature_dicts.append(features)
    X = vec.fit_transform(feature_dicts)
    return X, vec


def get_tweets_features(texts, tweets_texts, num_of_tweets, followers_count, listed_count, friends_count, default_profile_image, default_profile):
    count_mention = 0
    count_url = 0
    factor = 100
    features = {}

    for s in texts:
        if 'http' in s:
            count_url += 1
        if '@' in s:
            count_mention += 1
    if len(texts) == 0:
        features['tweets_avg_urls'] = 0
        features['tweets_avg_mentions'] = 0
    else:
        features['tweets_avg_urls'] = factor * count_url / len(texts)
        features['tweets_avg_mentions'] = factor * count_mention / len(texts)

    features['followers_count'] = followers_count
    features['listed_count'] = listed_count
    features['friends_count'] = friends_count
    features['default_profile_image'] = int(default_profile_image)
    features['default_profile'] = int(default_profile)

    # add the tri_gram feature
    tri_count_vec = CountVectorizer(min_df=1, max_df=1.0, ngram_range=(3, 3))
    try:
        user_words = tri_count_vec.fit_transform(tweets_texts)
    except ValueError:
        features['tri_gram_most_common'] = 0
        return features
    freqs = zip(tri_count_vec.get_feature_names(), user_words.sum(axis=0).tolist()[0])
    # sort from largest to smallest
    f_list = sorted(freqs, key=lambda x: -x[1])
    top_element = f_list[0]
    top_word = top_element[0]
    top_freq = top_element[1]
    # print(top_element)
    if num_of_tweets != 0:
        frequency = top_freq / num_of_tweets * 100
    else:
        frequency = 0
    features['tri_gram_most_common'] = frequency

    return features


def print_top_features(vec, count_vec, clf):
    # sort coefficients by class.
    features = vec.get_feature_names() + count_vec.get_feature_names()
    coef = [-clf.coef_[0], clf.coef_[0]]
    for ci, class_name in enumerate(clf.classes_):
        print('top 15 features for class %s:' % class_name)
        for fi in coef[ci].argsort()[-1:-16:-1]:  # descending order.
        # for fi in coef[ci].argsort()[-1:-56:-1]:  # descending order.
            print('%20s\t%.6f' % (features[fi], coef[ci][fi]))
        print()


# def print_top_features()
if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
