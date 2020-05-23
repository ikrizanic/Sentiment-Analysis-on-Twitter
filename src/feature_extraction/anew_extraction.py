import csv
import statistics
from nltk.stem import WordNetLemmatizer

anew = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/anew.csv"
lm = WordNetLemmatizer()
negations = ["no", "not", "n't"]

anew_dict = dict()
with open(anew) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        anew_dict.update({row["Word"]: [row["valence"], row["arousal"], row["dominance"]]})

def get_anew_scores(tokens):
    v_list = []  # holds valence scores
    a_list = []  # holds arousal scores
    d_list = []  # holds dominance scores
    neg = False

    for i in range(len(tokens)):
        token = lm.lemmatize(tokens[i])
        if i >= 3:
            if tokens[i - 1] in negations or tokens[i - 2] in negations or tokens[i - 3] in negations:
                neg = True
        if token in anew_dict.keys():
            values = anew_dict.get(token)
            v = float(values[0])
            a = float(values[1])
            d = float(values[2])

            if neg:
                v = 5 - (v - 5)
                a = 5 - (a - 5)
                d = 5 - (d - 5)

            v_list.append(v)
            a_list.append(a)
            d_list.append(d)

    return v_list, a_list, d_list

def get_anew_mean(tokens):
    v, a, d = get_anew_scores(tokens)
    if len(v) == 0:
        v.append(0)
    if len(a) == 0:
        a.append(0)
    if len(d) == 0:
        d.append(0)
    return statistics.mean(v), statistics.mean(a), statistics.mean(d)

def get_anew_median(tokens):
    v, a, d = get_anew_scores(tokens)
    if len(v) == 0:
        v.append(0)
    if len(a) == 0:
        a.append(0)
    if len(d) == 0:
        d.append(0)
    return statistics.median(v), statistics.median(a), statistics.median(d)

def get_anew_sum(tokens):
    v, a, d = get_anew_scores(tokens)
    return sum(v), sum(a), sum(d)
