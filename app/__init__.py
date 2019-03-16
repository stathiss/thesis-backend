import re
import tweepy
import json
import random
from flask import Flask, jsonify, request
from tweepy import OAuthHandler
from sources.utils import write_predictions, calculate_all_predictions

app = Flask(__name__)
app.config.from_object('config')


def tweet_to_full_text(tweet):
    if 'retweeted_status' in tweet:
        if 'full_text' in tweet['retweeted_status']:
            text = re.sub(r"http\S+", "", tweet['retweeted_status']['full_text'])
            text = re.sub(r"@\S+", "", text)
            return text.replace('\n', '')
        else:
            text = re.sub(r"http\S+", "", tweet['retweeted_status']['text'])
            text = re.sub(r"@\S+", "", text)
            return text.replace('\n', '')

    else:
            if 'full_text' in tweet:
                text = re.sub(r"http\S+", "", tweet['full_text'])
                text = re.sub(r"@\S+", "", text)
                return text.replace('\n', '')
            else:
                text = re.sub(r"http\S+", "", tweet['text'])
                text = re.sub(r"@\S+", "", text)
                return text.replace('\n', '')


@app.route('/', methods=["GET"])
def hello():
    hashtag = request.args.get('hashtag', '')

    consumer_key = app.config['TWITTER_CONSUMER_KEY']
    consumer_secret = app.config['TWITTER_CONSUMER_SECRET']
    access_token = app.config['TWITTER_ACCESS_TOKEN']
    access_secret = app.config['TWITTER_ACCESS_SECRET']

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

    trends = api.trends_place(1)
    public_tweets = tweepy.Cursor(api.search, q='#' + hashtag + ' ', lang='en', count=200, tweet_mode='extended').items(200)
    tweets = []
    ids = []
    texts = []
    counter = 1
    for tweet in public_tweets:
        full_tweet = tweet_to_full_text(tweet._json)
        # print(json.dumps(tweet._json, indent=2))
        if not any(d['text'] == full_tweet for d in tweets):
            ids.append(str(tweet.id))
            texts.append(full_tweet.encode('utf-8').strip())
            tweets.append({
                'text': full_tweet,
                'id': tweet.id_str,
                'counter': counter,
                'author': tweet.user.name,
                'date': tweet.created_at,
                'regression': {
                    'fear': random.random(),
                    'joy': random.random(),
                    'anger': random.random(),
                    'sadness': random.random()},
                'ordinal': {
                    'fear': random.uniform(0, 3),
                    'joy': random.uniform(0, 3),
                    'anger': random.uniform(0, 3),
                    'sadness': random.uniform(0, 3)
                }})
            counter = counter + 1
            if counter > 100:
                break
    write_predictions('test_tweets.txt',
                      [ids, texts, ['emotion' for _ in range(len(ids))]], [0 for _ in range(len(ids))])
    predictions = calculate_all_predictions(tweets)
    my_json = jsonify({
        'trends': trends[0]['trends'],
        'tweets': tweets,
        'top': {
            'joy': {
                'tweet': 'Hello Darkness my old friend',
                'intensity': 0.76
            },
            'fear': {
                'tweet': 'Hello Darkness my old friend',
                'intensity': 0.76
            },
            'sadness': {
                'tweet': 'Hello Darkness my old friend',
                'intensity': 0.76
            },
            'anger': {
                'tweet': 'Hello Darkness my old friend',
                'intensity': 0.76
            },
        }
    })
    my_json.headers.add('Access-Control-Allow-Origin', '*')
    return my_json
# Import the routes.
