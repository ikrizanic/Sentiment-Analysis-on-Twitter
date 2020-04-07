import nltk
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import words

sp = SpellCorrector(corpus="english")
nltk.download("words")
words = set(words.words())

def spell_check(raw):
    correct_raw = list()
    for word in raw.split(" "):
        if word in words:
            correct_raw.append(word)
        else:
            correct_raw.append(sp.correct(word))
    return " ".join(correct_raw)
