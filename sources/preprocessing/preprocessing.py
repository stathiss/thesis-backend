# -*- coding: utf-8 -*-
# from tweetokenize import Tokenizer


def split_hashtags(hashtag):
    if hashtag[0] == '#':
        hashtag = hashtag[1:]
        return hashtag.split('_')
    return [hashtag]
