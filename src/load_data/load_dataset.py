import spacy
from podium.storage import Vocab, Field, LabelField
from podium.datasets import TabularDataset

from src.hooks.spell_check import *
from src.hooks.annotation_normalization import *
from src.hooks.pretokenization import *

def make_labels(dataset):
    labels = list()
    for lab in [getattr(d, "label")[0] for d in dataset]:
        if lab == "positive":
            labels.append(2)
        elif lab == "neutral":
            labels.append(1)
        else:
            labels.append(0)
    return labels

def print_data(dataset):
    for d in dataset:
        print(d.tweet[1])
        print(len(d.tweet[1]))
        print(d.pos[1])
        print(len(d.pos[1]))
        print(d.dep[1])
        print(len(d.dep[1]))
        print(d.anot)
        dobj, amod, nsubj = [], [], []
        for i in range(len(d.tweet[1])):
            if d.dep[1][i] == "dobj":
                dobj.append(d.tweet[1][i])
            elif d.dep[1][i] == "amod":
                amod.append(d.tweet[1][i])
            elif d.dep[1][i] == "nsubj":
                nsubj.append(d.tweet[1][i])
        print(dobj)
        print(amod)
        print(nsubj)
        print("-" * 50)


def load_dataset(dataset_path):
    nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])
    nlp_full = spacy.load('en_core_web_lg')

    def extract_text_hook(raw, tokenized):
        return raw, [token.text for token in tokenized]

    def extract_pos_hook(raw, tokenized):
        return raw, [token.pos_ for token in tokenized]

    def extract_dep_hook(raw, tokenized):
        return raw, [token.dep_ for token in tokenized]

    tweet = Field(name='tweet', vocab=Vocab(), store_as_raw=True, tokenizer=nlp)
    anot = Field(name="anot", vocab=Vocab(), store_as_raw=True)
    pos = Field(name='pos', vocab=Vocab(), tokenizer=nlp)
    dep = Field(name='dep', vocab=Vocab(), tokenizer=nlp_full)
    label = LabelField(name='label')

    tweet.add_pretokenize_hook(repair_chars)
    tweet.add_pretokenize_hook(remove_usernames)
    tweet.add_pretokenize_hook(remove_links)
    tweet.add_pretokenize_hook(remove_punctuation)
    pos.add_pretokenize_hook(repair_chars)
    pos.add_pretokenize_hook(remove_punctuation)
    pos.add_pretokenize_hook(remove_usernames)
    pos.add_pretokenize_hook(remove_links)
    dep.add_pretokenize_hook(repair_chars)
    dep.add_pretokenize_hook(remove_punctuation)
    dep.add_pretokenize_hook(remove_usernames)
    dep.add_pretokenize_hook(remove_links)

    anot.add_pretokenize_hook(repair_chars)
    anot.add_pretokenize_hook(annotation_normalization)

    tweet.add_posttokenize_hook(extract_text_hook)
    pos.add_posttokenize_hook(extract_pos_hook)
    anot.add_posttokenize_hook(spell_check_tokens)
    anot.add_posttokenize_hook(replace_slang)
    dep.add_posttokenize_hook(extract_dep_hook)

    fields = {'text': (tweet, anot, pos, dep), 'label': label}

    print("Loading dataset from: " + dataset_path + "\n")
    return TabularDataset(dataset_path, format='tsv', fields=fields)

