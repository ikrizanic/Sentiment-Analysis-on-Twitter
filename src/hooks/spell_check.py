import csv
import nltk
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import words


with open('/home/ikrizanic/pycharm/zavrsni/data/slang.csv', mode='r') as infile:
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


def spell_check(raw):
    correct_raw = list()
    for word in raw.split(" "):
        if word in words:
            correct_raw.append(word)
        else:
            correct_raw.append(sp.correct(word))
    return " ".join(correct_raw)


def spell_check_tokens(raw, tokens):
    correct_tokens = list()
    for token in tokens:
        if token[0] == "<":
            continue
        if token[0] in punctuations:
            continue
        if token in words:
            correct_tokens.append(token)
        else:
            correct_tokens.append(sp.correct(token))
    return raw, correct_tokens
