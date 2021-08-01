import nltk
from nltk.text import TextCollection
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import os
import string
import json
import pickle
import time
import gensim
from gensim import corpora, models, similarities
from tqdm import tqdm


# 对词性进行还原，例如复数变成单数
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def for_keywords_change_word_by_lemmatize(doc):
    """词性还原"""
    word_list = nltk.word_tokenize(doc)
    return [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w)) for w in word_list]


def for_keywords_remove_stopword(doc):
    word_split = doc
    valid_word = []
    for word in word_split:
        word = word.strip().strip(string.digits)
        if word  != "":
            valid_word.append(word)
    word_split = valid_word
    stop_words = set(stopwords.words('english'))
    # add punctuations
    punctuations = list(string.punctuation)
    [stop_words.add(punc) for punc in punctuations]
    # remove null
    stop_words.add("null")

    return [word for word in word_split if word not in stop_words]


def transform_sentence(doc):
    if doc is None:
        return ""
    doc = doc.strip().replace('_', '').replace('-', '').replace('/sub', '').replace(' sub ', " ")
    doc = doc.replace("ABSTRACTS", '')
    doc = for_keywords_change_word_by_lemmatize(doc)
    doc = for_keywords_remove_stopword(doc)
    return doc


def transform_pub(docs):
    """注意，这个函数会改变输入的字典值"""
    new_docs = docs
    for data in tqdm(docs, desc="transform publications"):
        # 处理标题：词干化->去掉停用词->加入if-idf
        if 'title' in docs[data]:
            new_docs[data]['title'] = transform_sentence(docs[data]['title'])
        else:
            new_docs[data]['title'] = []

        # 处理关键词：词干化
        new_keywords_list = []

        if 'keywords' in docs[data]:
            new_keywords_list = [for_keywords_change_word_by_lemmatize(
                keywords) for keywords in docs[data]['keywords']]
        else:
            new_keywords_list = []
        new_docs[data]['keywords'] = new_keywords_list

        # 处理摘要: 词干化
        if 'abstract' in docs[data]:
            new_docs[data]['abstract'] = transform_sentence(docs[data]['abstract'])
        else:
            new_docs[data]['abstract'] = []
    return new_docs