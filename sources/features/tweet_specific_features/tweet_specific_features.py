import re
from hunspell import Hunspell
from sources.loaders.loaders import parse_dataset
import emoji
import string


def number_of_emojis(tweet):
    counter = 0
    for word in tweet.split():
        if emoji.demojize(word) != word:
            counter += 1
    return counter


def number_of_syntax_errors(tweet):
    counter = 0
    h = Hunspell()
    for word in tweet.split():
        if word.isalpha() and word.islower():
            if not h.spell(word) and h.suggest(word):
                counter += 1
    return counter


def parse_tweet_specific_features(task, emotion, label):
    repeating_re = re.compile(r"([a-zA-Z])\1\1+")
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    tweets = parse_dataset(task, emotion, label)[1]
    print('length', len(tweets))
    features = []
    for tweet in tweets:
        current_feature = list()
        # 1. Length of tweet
        current_feature.append(len(tweet))
        # 2. Number of words
        current_feature.append(len(tweet.split()))
        # 3. Number of emojis
        current_feature.append(number_of_emojis(tweet))
        # 4. Number of continuous letters (eg looove)
        current_feature.append(len(re.findall(repeating_re, tweet)))
        # 5. Number punctuation mark
        current_feature.append(count(tweet, string.punctuation))
        # 6. syntax errors
        current_feature.append(number_of_syntax_errors(tweet))
        # 7. words in uppercase
        current_feature.append(sum(1 for c in tweet if c.isupper()))
        features.append(current_feature)
