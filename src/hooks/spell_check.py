import csv
import nltk
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import words

#  CHANGE PATH FOR SERVER
local = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/slang.csv"
djurdja = '/home/ikrizanic/pycharm/zavrsni/data/slang.csv'
with open(local, mode='r') as infile:
    reader = csv.reader(infile, delimiter=';')
    slang_dict = dict(reader)

sp = SpellCorrector(corpus="english")
nltk.download("words")
words = set(words.words())
punctuations = '''!()-[]{};:\'"\,<>./?@#$%^&*_~'''


def replace_slang(raw, tokenized):
    tokens = []
    for token in tokenized:
        if token not in words:
            for key, value in slang_dict.items():
                if str(key).lower() == str(token).lower():
                    token = value.split(" ")
            if type(token) is list:
                tokens.extend(token)
            else:
                tokens.append(token)
        else:
            tokens.append(token)
    return raw, tokens

def replace_slang_tokens(tokenized):
    tokens = []
    for token in tokenized:
        if token not in words:
            for key, value in slang_dict.items():
                if str(key).lower() == str(token).lower():
                    token = value.split(" ")
            if type(token) is list:
                tokens.extend(token)
            else:
                tokens.append(token)
        else:
            tokens.append(token)
    return tokens

def spell_check(raw):
    correct_raw = list()
    for word in raw.split(" "):
        if word in words:
            correct_raw.append(word)
        else:
            correct_raw.append(sp.correct(word))
    return " ".join(correct_raw)


def spell_check_tokens(tokens):
    correct_tokens = list()
    for token in tokens:
        if token.strip()[0] == "<":
            continue
        if token.strip()[0] in punctuations:
            continue
        if token.strip() in words:
            correct_tokens.append(token.strip())
        else:
            correct_tokens.append(sp.correct(token.strip()))
    return correct_tokens
