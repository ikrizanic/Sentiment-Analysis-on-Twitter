import spacy
from podium.storage import Vocab, Field, LabelField, MultioutputField
from podium.datasets import TabularDataset

from spell_check import *
from annotation_normalization import *
from pretokenization import *


def load_dataset(dataset_path):
    nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])

    def extract_text_hook(raw, tokenized):
        return raw, [token.text for token in tokenized]

    def extract_pos_hook(raw, tokenized):
        return raw, [token.pos_ for token in tokenized]

    tweet = Field(name='tweet', vocab=Vocab(), store_as_raw=True, tokenizer=nlp)
    anot = Field(name="anot", vocab=Vocab(), store_as_raw=True)
    pos = Field(name='pos', vocab=Vocab(), tokenizer=nlp)
    label = LabelField(name='label')

    tweet.add_pretokenize_hook(repair_chars)
    tweet.add_pretokenize_hook(remove_usernames)
    tweet.add_pretokenize_hook(remove_links)

    anot.add_pretokenize_hook(repair_chars)
    anot.add_pretokenize_hook(annotation_normalization)

    tweet.add_posttokenize_hook(extract_text_hook)
    pos.add_posttokenize_hook(extract_pos_hook)
    anot.add_posttokenize_hook(spell_check_tokens)

    fields = {'text': (tweet, anot, pos), 'label': label}

    print("Loading dataset from: " + dataset_path + "\n")
    return TabularDataset(dataset_path, format='tsv', fields=fields)

