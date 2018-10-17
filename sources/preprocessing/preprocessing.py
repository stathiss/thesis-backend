# -*- coding: utf-8 -*-
from tweetokenize import Tokenizer
from hunspell import Hunspell
from sources.loaders.loaders import parse_dataset
import emoji


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


def tweet_tokenizer(task, emotion, label):
    token = Tokenizer(normalize=2)
    h = Hunspell()
    tweets = parse_dataset(task, emotion, label)[1]
    for tweet in tweets:
        print('tweet:')
        tweet = tweet.replace('\\n', '')
        print(tweet)
        print('tokenizaton:')
        tokens = token.tokenize(tweet)
        print(tokens)
        spell_check = []
        print('spell check')
        for word in tokens:
            if word.isalpha() and word.islower():
                if not h.spell(word) and h.suggest(word):
                    spell = h.suggest(word)
                    spell = spell[0].split(' ')
                    spell = list(map(lambda x: x.lower(), spell))
                    spell_check.extend(spell)
                else:
                    spell_check.append(word)
            else:
                spell_check.append(word)
        print(spell_check)
        print('Demojize: ')
        demojize = []
        for word in spell_check:
            demojize.extend(emoji_to_description(word))
        print(demojize)
        print('\n')
        print('\n')
