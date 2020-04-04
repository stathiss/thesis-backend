import re
import tweepy
import random
import datetime
import pymongo
import numpy as np
from flask import Flask, jsonify, request
from tweepy import OAuthHandler
from sources.utils import write_predictions, calculate_all_predictions
from mongokit import Connection, Document

app = Flask(__name__)
app.config.from_object('config')
consumer_key = app.config['TWITTER_CONSUMER_KEY']
consumer_secret = app.config['TWITTER_CONSUMER_SECRET']
access_token = app.config['TWITTER_ACCESS_TOKEN']
access_secret = app.config['TWITTER_ACCESS_SECRET']

connection = Connection(app.config['MONGODB_HOST'],
                        app.config['MONGODB_PORT'])

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


def tweet_to_full_text(tweet):
    if 'retweeted_status' in tweet:
        if 'full_text' in tweet['retweeted_status']:
            text = re.sub(r"http\S+", "(link)", tweet['retweeted_status']['full_text'])
            text = re.sub(r"@\S+", "", text)
            return text.replace('\n', '')
        else:
            text = re.sub(r"http\S+", "(link)", tweet['retweeted_status']['text'])
            text = re.sub(r"@\S+", "", text)
            return text.replace('\n', '')

    else:
            if 'full_text' in tweet:
                text = re.sub(r"http\S+", "(link)", tweet['full_text'])
                text = re.sub(r"@\S+", "", text)
                return text.replace('\n', '')
            else:
                text = re.sub(r"http\S+", "(link)", tweet['text'])
                text = re.sub(r"@\S+", "", text)
                return text.replace('\n', '')


@app.route('/', methods=["GET"])
def get_predictions():
    # Search for tweets with specific hashtag
    hashtag = request.args.get('hashtag', '')
    max_tweets = int(request.args.get('number', '100'))
    public_tweets = tweepy.Cursor(api.search,
                                  q=hashtag,
                                  lang='en',
                                  count=200,
                                  tweet_mode='extended').items(200)
    
    # Add it to database if it does not exist
    hashtags = connection['tweet-ai'].hashtags
    result = hashtags.find_one({'hashtag': hashtag})
    if not result and hashtag and hashtag != '#':
        hashtags.insert({'hashtag': hashtag,
                         'createdAt': datetime.datetime.now(),
                         'updatedAt': datetime.datetime.now()})
    else:
        hashtags.update({'hashtag': hashtag}, {'$set': {'updatedAt': datetime.datetime.now()}})
    tweets = []
    ids = []
    texts = []
    counter = 1
    for tweet in public_tweets:
        full_tweet = tweet_to_full_text(tweet._json)
        if not any(d['text'] == full_tweet for d in tweets):
            ids.append(str(tweet.id))
            texts.append(full_tweet.encode('utf-8').strip())
            tweets.append({
                'text': full_tweet,
                'id': tweet.id_str,
                'counter': counter,
                'author': tweet.user.name,
                'screen_name': tweet.user.id,
                'date': tweet.created_at,
                'regression': {
                    'fear': random.random(),
                    'joy': random.random() * 3,
                    'anger': random.random(),
                    'sadness': random.random()},
                })
            counter = counter + 1
            if counter > max_tweets:
                break
    write_predictions('test_tweets.txt',
                      [ids, texts, ['emotion' for _ in range(len(ids))]], [0 for _ in range(len(ids))])
    tweets, top_tweets_indexes, averages, ordinal_class, e_c = calculate_all_predictions(tweets)
    print(ordinal_class)
    my_json = jsonify({
        'e_c': e_c,
        'averages': {
            'anger': np.round(averages[0], 2),
            'fear': np.round(averages[1], 2),
            'joy': np.round(averages[2], 2),
            'sadness': np.round(averages[3], 2)
        },
        'tweets': tweets,
        'top': {
            'anger': {
                'tweet': tweets[top_tweets_indexes[0]]['text'],
                'intensity': tweets[top_tweets_indexes[0]]['regression']['anger'],
                'id': tweets[top_tweets_indexes[0]]['id']
            },
            'fear': {
                'tweet': tweets[top_tweets_indexes[1]]['text'],
                'intensity': tweets[top_tweets_indexes[1]]['regression']['fear'],
                'id': tweets[top_tweets_indexes[1]]['id']
            },
            'joy': {
                'tweet': tweets[top_tweets_indexes[2]]['text'],
                'intensity': tweets[top_tweets_indexes[2]]['regression']['joy'],
                'id': tweets[top_tweets_indexes[2]]['id']
            },
            'sadness': {
                'tweet': tweets[top_tweets_indexes[3]]['text'],
                'intensity': tweets[top_tweets_indexes[3]]['regression']['sadness'],
                'id': tweets[top_tweets_indexes[3]]['id']
            }
        },
        'ordinal_class': {
            'anger': ordinal_class[0],
            'fear': ordinal_class[1],
            'joy': ordinal_class[2],
            'sadness': ordinal_class[3]
        }
    })
    my_json.headers.add('Access-Control-Allow-Origin', '*')
    return my_json


@app.route('/trends', methods=["GET"])
def get_trends():
    trends = api.trends_place(1)[0]['trends']
    english_trends = []
    for trend in trends:
        try:
            trend['name'].encode('utf-8').decode('ascii')
            english_trends.append(trend)
        except UnicodeError:
            print(trend['name'])
    my_json = jsonify({
        'trends': english_trends
    })
    my_json.headers.add('Access-Control-Allow-Origin', '*')
    return my_json


@app.route('/searches', methods=["GET"])
def get_searches():
    hashtags = connection['tweet-ai'].hashtags
    results = hashtags.find({}).sort('updatedAt', pymongo.DESCENDING).limit(15)
    return_value = []
    for result in results:
        return_value.append(result['hashtag'])
    my_json = jsonify({
        'searches': [{'name': value} for value in return_value]
    })
    my_json.headers.add('Access-Control-Allow-Origin', '*')
    return my_json

