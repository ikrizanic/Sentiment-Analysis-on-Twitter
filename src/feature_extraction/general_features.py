from src.feature_extraction.anew_extraction import *
import numpy as np

def calculate_anew(tokens, mean=True, median=True):
    if mean:
        if median:
            return get_anew_mean(tokens) + get_anew_median(tokens)
        else:
            return get_anew_mean(tokens)
    else:
        if median:
            return get_anew_median(tokens)
        else:
            print("Nothing to return!")

def get_anew(text_tokens, mean=True, median=True, sum=False):
    features = [list() for i in range(len(text_tokens))]
    for i in range(len(text_tokens)):
        features[i].extend((calculate_anew(text_tokens[i], mean, median)))
        if sum:
            features[i].extend(get_anew_sum(text_tokens[i]))
    return features

def get_anew_full(text_tokens, max_size=15):
    features = [list() for i in range(len(text_tokens))]
    for i in range(len(text_tokens)):
        x, y, z = get_anew_scores(text_tokens[i])
        for j in range(max_size):
            if j < len(x):
                features[i].append(x[j])
                features[i].append(y[j])
                features[i].append(z[j])
            else:
                features[i].extend([0, 0, 0])
    return features
