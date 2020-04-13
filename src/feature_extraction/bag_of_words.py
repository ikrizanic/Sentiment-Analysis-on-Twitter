import heapq
import nltk
import numpy as np
nltk.download("punkt")

def bag_of_words(dataset):
    words_count = {}
    for sent in dataset:
        for word in sent:
            if word not in words_count.keys():
                words_count[word] = 1
            else:
                words_count[word] += 1

    freq_words = heapq.nlargest(500, words_count,  key=words_count.get)

    features = []
    for data in dataset:
        vector = []
        for word in freq_words:
            if word in data:
                vector.append(1)
            else:
                vector.append(0)
        features.append(vector)
    return np.asarray(features)
