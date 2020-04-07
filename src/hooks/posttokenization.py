from nltk.corpus import stopwords


def remove_stopwords(raw, tokenized):
    stop_words_set = set(stopwords.words('english'))
    tokens = []
    for token in tokenized:
        token_lower = token if token.islower() else token.lower()
        if token_lower not in stop_words_set:
            tokens.append(token)
    return raw, tokens