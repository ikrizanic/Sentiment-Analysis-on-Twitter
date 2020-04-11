import nltk
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import words

sp = SpellCorrector(corpus="english")
nltk.download("words")
words = set(words.words())
punctuations = '''!()-[]{};:\'"\,<>./?@#$%^&*_~'''


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
