from src.feature_extraction.anew_extraction import *


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

def get_anew(text_tokens, mean=True, median=True, sum=True):
    features = [list() for i in range(len(text_tokens))]
    for i in range(len(text_tokens)):
        features[i].extend(calculate_anew(text_tokens[i], mean, median))
        if sum:
            features[i].extend(get_anew_sum(text_tokens[i]))
    return features

