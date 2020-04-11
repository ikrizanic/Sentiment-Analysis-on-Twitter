import re

def has_all_caps(dataset, features):
    for i in range(len(dataset)):
        tweet = dataset[i].anot[0]
        reg = re.findall(r'<allcaps>(.*)</allcaps>', tweet)
        if len(reg) != 0:
            # 1 if there is all caps, 0 if there is none
            features[i].append(1)
            words = 0
            for r in reg:
                words += len(r.split(" "))
            # number of words in all caps
            features[i].append(words)
        else:
            features[i].append(0)
            features[i].append(0)

def has_hash_tag(dataset, features):
    for i in range(len(dataset)):
        tweet = dataset[i].anot[0]
        reg = re.findall(r'<hashtag>([^<]*)</hashtag>', tweet)
        if len(reg) != 0:
            # 1 if there is hash tag, 0 if there is none
            features[i].append(1)
            # number of hash tags
            features[i].append(len(reg))
        else:
            features[i].append(0)
            features[i].append(0)

def has_normalizations(dataset, features):
    for i in range(len(dataset)):
        tweet = dataset[i].anot[0]
        features[i].append(len(re.findall(r'<url>', tweet)))
        features[i].append(len(re.findall(r'<email>', tweet)))
        features[i].append(len(re.findall(r'<percent>', tweet)))
        features[i].append(len(re.findall(r'<money>', tweet)))
        features[i].append(len(re.findall(r'<phone>', tweet)))
        features[i].append(len(re.findall(r'<user>', tweet)))
        features[i].append(len(re.findall(r'<time>', tweet)))
        features[i].append(len(re.findall(r'<date>', tweet)))
        features[i].append(len(re.findall(r'<number>', tweet)))

def has_annotations(dataset, features):
    for i in range(len(dataset)):
        tweet = dataset[i].anot[0]
        features[i].append(len(re.findall(r'<elongated>', tweet)))
        features[i].append(len(re.findall(r'<emphasis>', tweet)))
        features[i].append(len(re.findall(r'<repeated>', tweet)))
        features[i].append(len(re.findall(r'<censored>', tweet)))


def extract_boolean_features(dataset):
    print("Extracting boolean features...\n")
    features = [list() for i in range(len(dataset))]
    has_all_caps(dataset, features)
    has_hash_tag(dataset, features)
    has_normalizations(dataset, features)
    has_annotations(dataset, features)
    return features

def count_words(dataset, features):
    for i in range(len(dataset)):
        features[i].append(len(dataset[i].tweet[1]))

