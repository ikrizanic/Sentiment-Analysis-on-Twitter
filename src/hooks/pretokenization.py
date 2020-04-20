import re
import pandas as pd
import string


def remove_punctuation(raw):
    return raw.translate(str.maketrans('', '', string.punctuation))

def remove_links(raw):
    raw = re.sub(r'http.*\b', '', raw)
    return raw


# some chars are lost in reading of dataset
def repair_chars(raw):
    raw = re.sub(r'\\u2019', "\'", raw)
    raw = re.sub(r'\\u002c', ',', raw)
    raw = re.sub(r'&lt', '>', raw)
    raw = re.sub(r'&gt', '<', raw)
    raw = re.sub(r'&amp;', '&', raw)
    return raw


# user names are parts of text starting witt @
def remove_usernames(raw):
    raw = re.sub(r'@[^\s]*', '', raw)
    return raw


def read_raw_tweets(data_path, sep="\t"):
    raw_data_merge = pd.read_csv(data_path, sep=sep, names=["label", "text"])
    clean_data = dict()
    for i in (range(len(raw_data_merge))):
        tweet = raw_data_merge.text[i]
        if tweet not in clean_data and isinstance(tweet, str):
            tweet = repair_chars(tweet)
            tweet = remove_usernames(tweet)
            tweet = remove_links(tweet)
            clean_data.update({tweet: raw_data_merge.label[i]})

    tweets = list()
    polarities = list()

    for text in clean_data.keys():
        tweets.append(text)

    for polarity in clean_data.values():
        polarities.append(polarity)

    return tweets, polarities


EMOTICONS, useful_emoticons = dict(), dict()
def find_useful_emoticons(emoticons_path, data_path):
    tweets, polarities = read_raw_tweets(data_path)
    emoticons_file = pd.read_csv(emoticons_path, sep="  ->  ", names=["emoji", "meaning"])

    for i in range(len(emoticons_file)):
        e = emoticons_file.emoji[i]
        m = emoticons_file.meaning[i]
        EMOTICONS.update({e: m})

    distribution = dict()
    count = 0
    [distribution.update({emot: [0, 0, 0, 0]}) for emot in EMOTICONS.keys()]
    for i in range(len(tweets)):
        flag = 0
        for emot in EMOTICONS:
            if tweets[i].find(emot) != -1:
                flag = 1
                distribution[emot][3] += 1
                #                 print(emot + "   ->   "  + dataset[i].text[0])
                if polarities[i] == "positive":
                    distribution[emot][0] += 1
                elif polarities[i] == "neutral":
                    distribution[emot][1] += 1
                else:
                    distribution[emot][2] += 1

        if flag != 1:
            count += 1
    print("Sentences without emoticons: " + str(count * 100 / len(tweets)) + "%")

    emoticons_score = dict()

    for k, v in distribution.items():
        if v[3] < 5:
            v[3] *= 3
        if v[3] == 0:
            score = 0
        else:
            score = (v[0] - v[2]) / v[3]
        emoticons_score.update({k: score})

    for k, v in emoticons_score.items():
        if v != 0:
            useful_emoticons.update({k: v})


def replace_unuseful_emoticons(raw):
    for k, v in EMOTICONS.items():
        if k not in useful_emoticons.keys() and k in raw:
            raw.replace(k, v)
    return raw




