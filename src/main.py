import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import re
from src.hooks.pretokenization import *
from src.hooks.spell_check import *
from src.load_data.load_dataset import *

# NLTK
import nltk
import codecs
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")


def main():
    find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/merge_bez2016.csv")
    dataset = load_dataset("~/pycharm/zavrsni/data/merge_bez2016.csv")
    [print(d + "\n") for d in dataset]


if __name__ == '__main__':
    main()
