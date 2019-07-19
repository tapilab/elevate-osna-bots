# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import sys
import gzip
import json
import pandas as pd
import re
from . import credentials_path, config
from collections import Counter


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


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
