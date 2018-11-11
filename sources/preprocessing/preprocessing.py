# -*- coding: utf-8 -*-
from tweetokenize import Tokenizer
# noinspection PyUnresolvedReferences
from hunspell import Hunspell
from sources.loaders.loaders import parse_dataset
import emoji


def tokens_to_sentence(tokens):
    sentence = u''
    for token in tokens[:-1]:
        sentence += token + u' '
    sentence += tokens[-1]
    return sentence


def split_hashtags(hashtag):
    if hashtag[0] == '#':
        hashtag = hashtag[1:]
        return hashtag.split('_')
    return [hashtag]


def emoji_to_description(word):
    if emoji.demojize(word) != word:
        return emoji.demojize(word)[1:-1].split('_')
    else:
        return [word]


def spell_check(tweet):
    words = 0
    mistakes = 0
    h = Hunspell()
    print('spell check')
    for word in tweet:
        if word.isalpha() and word.islower():
            words = words + 1
            if not h.spell(word) and h.suggest(word):
                mistakes = mistakes + 1
                # spell = h.suggest(word)
                # spell = spell[0].split(' ')
                # spell = list(map(lambda x: x.lower(), spell))
                # spell_check.extend(spell)
    print(words, mistakes, mistakes/float(words) if words != 0 else None)


def tweet_tokenizer(task, emotion, label):
    tokenized_tweets = []
    token = Tokenizer(normalize=2)
    tweets = parse_dataset(task, emotion, label)[1]
    print('length', len(tweets))
    for tweet in tweets:
        print('tweet:')
        tweet = tweet.replace('\\n', '')
        print(tweet)
        print('tokenizaton:')
        tokens = token.tokenize(tweet)
        print(tokens)
        print('Demojize: ')
        demojize = []
        for word in tokens:
            demojize.extend(emoji_to_description(word))
        print(demojize)
        tokenized_tweets.append(tokens_to_sentence(demojize))
    print('Tokenized tweets:')

    print(tokenized_tweets)
    return tokenized_tweets
    # spell_check(tokens)
