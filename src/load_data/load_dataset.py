from podium.datasets import Iterator
from podium.storage import Vocab, Field, LabelField, MultioutputField
from podium.storage.vectorizers.tfidf import CountVectorizer
from podium.datasets import TabularDataset
import functools
import spacy

def load_dataset(dataset_path):
    print("Loading dataset from: " + dataset_path + "\n")
    nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])

    def extract_text_hook(raw, tokenized):
        return raw, [token.text for token in tokenized]

    def extract_pos_hook(raw, tokenized):
        return raw, [token.pos_ for token in tokenized]

    def extract_vec_hook(raw, tokenized):
        return raw, [token.vector_norm for token in tokenized]

    text = Field(name='text', vocab=Vocab(), store_as_raw=True)
    text.add_posttokenize_hook(extract_text_hook)

    pos = Field(name='pos', vocab=Vocab())
    pos.add_posttokenize_hook(extract_pos_hook)

    # vec = Field(name='vec')
    # vec.add_posttokenize_hook(extract_vec_hook)

    text = MultioutputField([text, pos], tokenizer=nlp)

    label = LabelField(name='label')
    fields = {'text': text, 'label': label}

    return TabularDataset(dataset_path, format='tsv', fields=fields)

