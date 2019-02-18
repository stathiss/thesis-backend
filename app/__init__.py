import re
import config
import tweepy
import json
from flask import Flask, jsonify, request
from tweepy import OAuthHandler

app = Flask(__name__)


def tweet_to_full_text(tweet):
    if 'retweeted_status' in tweet:
        if 'full_text' in tweet['retweeted_status']:
            return re.sub(r"http\S+", "", tweet['retweeted_status']['full_text'])
        else:
            return re.sub(r"http\S+", "", tweet['retweeted_status']['text'])

    else:
        if 'full_text' in tweet:
            return re.sub(r"http\S+", "", tweet['full_text'])
        else:
            return re.sub(r"http\S+", "", tweet['text'])


@app.route('/', methods=["GET"])
def hello():
    hashtag = request.args.get('hashtag', '')

    consumer_key = config.TWITTER_CONSUMER_KEY
    consumer_secret = config.TWITTER_CONSUMER_SECRET
    access_token = config.TWITTER_ACCESS_TOKEN
    access_secret = config.TWITTER_ACCESS_SECRET

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

    public_tweets = tweepy.Cursor(api.search, q='#' + hashtag, lang='en', count=200, tweet_mode='extended').items(200)
    tweets = []
    ids = []
    texts = []
    for tweet in public_tweets:
        # json_tweet = tweet._json
        full_tweet = tweet_to_full_text(tweet._json)
        print(json.dumps(tweet._json, indent=2))
        if not any(d['text'] == full_tweet for d in tweets):
            tweets.append({
                'text': full_tweet,
                'id': tweet.id})
        print(tweet.full_text)
        print('\n')

    my_json = jsonify(tweets)
    my_json.headers.add('Access-Control-Allow-Origin', '*')
    return my_json
# Import the routes.
