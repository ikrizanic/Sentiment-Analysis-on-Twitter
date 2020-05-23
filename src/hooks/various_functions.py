from nltk.stem import WordNetLemmatizer
import spacy
from src.hooks.pretokenization import *
from src.hooks.posttokenization import *
from src.hooks.spell_check import *
from src.hooks.annotation_normalization import *
import numpy as np
from sklearn.model_selection import train_test_split
import copy

nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])

def tokenize(raw, tokenizer="split"):
    if tokenizer == "spacy":
        return [token.text for token in nlp.tokenizer(raw)]
    if tokenizer == "split":
        return raw.split(" ")


lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    lem = list()
    for token in tokens:
        lem.append(lemmatizer.lemmatize(token))
    return lem


def build_vocab(data):
    vocab = dict()
    index = 1
    for sent in data:
        for word in sent:
            if word not in vocab.keys():
                vocab.update({word: index})
                index += 1
    return vocab

def encode_sentence(sentence, vocab):
    encoded = list()
    for word in sentence:
        if word in vocab.keys():
            encoded.append(vocab[word])
        else:
            encoded.append(0)
    return encoded


def encode_data(data, vocab):
    encoded_data = list()
    for sent in data:
        encoded_data.append(encode_sentence(sent, vocab))
    return encoded_data


def split_train_validate_test(data, labels, train_valtest_ratio, validate_test_ratio, random_state=42):
    X_train, X_valtest, y_train, y_valtest = train_test_split(data, labels, test_size=train_valtest_ratio,
                                                              random_state=random_state)
    X_validate, X_test, y_validate, y_test = train_test_split(X_valtest, y_valtest, test_size=validate_test_ratio,
                                                              random_state=random_state)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def process_dataset(data):
    dataset = list()
    for i in tqdm(range(len(data))):
        new_tweet = repair_chars(data[i])
        anot = copy.deepcopy(new_tweet)
        anot = new_tweet
        new_tweet = remove_usernames(new_tweet)
        new_tweet = remove_links(new_tweet)
        new_tweet = remove_punctuation(new_tweet)

        tweet_tokens = tokenize(new_tweet, tokenizer="spacy")
        tweet_tokens = remove_stopwords(raw="", tokenized=tweet_tokens)

        anot = annotation_normalization(anot)
        anot_tokens = tokenize(anot, tokenizer="split")
        anot_tokens = spell_check_tokens(anot_tokens)
        anot_tokens = replace_slang_tokens(anot_tokens)
        anot_tokens = remove_stopwords_tokens(anot_tokens)
        anot_tokens = lemmatize(anot_tokens)

        dataset.append({"tweet": new_tweet, "tweet_tokens": tweet_tokens, "anot": anot, "anot_tokens": anot_tokens})
        # dataset.append({"anot_tokens": anot_tokens})
    return dataset


def create_vocab_encode_data(tokens):
    vocab = build_vocab(tokens)
    encoded_data = encode_data(data=tokens, vocab=vocab)
    return vocab, encoded_data


def pad_encoded_data(encoded, seq_length):
    features = np.zeros((len(encoded), seq_length), dtype=float)
    for i, review in enumerate(encoded):
        if len(review) > seq_length:
            review = review[:seq_length]
        zeroes = list(np.zeros(seq_length - len(review)))
        new = zeroes + review
        features[i, :] = np.array(new)
    return features


def make_embedding_matrix(vocab, embedding_dim=300):
    hits, misses = 0, 0
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for word, i in vocab.items():
        token = nlp(word)
        if token.has_vector:
            embedding_matrix[i] = token.vector
            hits += 1
        else:
            misses += 1

    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix
