import warnings
import nltk

from boolean_features import extract_boolean_features
from general_features import get_anew
from load_dataset import load_dataset, find_useful_emoticons
from models import *
from read_features import read_features

nltk.download("wordnet")
warnings.simplefilter('ignore')


def main():
    find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/main_data.csv")
    dataset = load_dataset("~/pycharm/zavrsni/data/main_data.csv")
    # features = [list() for i in range(len(dataset))]
    # boolean_features = extract_boolean_features(dataset)
    # general_features = get_anew([getattr(data, "anot")[1] for data in dataset])
    #
    # for i in range(len(dataset)):
    #     features[i].extend(boolean_features[i])
    #     features[i].extend(general_features[i])

    # file = open("/home/ikrizanic/pycharm/zavrsni/data/test_features.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(features[i]))
    #     file.write("\n")
    features = read_features("/home/ikrizanic/pycharm/zavrsni/data/features.txt")

    print(len(features))
    labels = list()
    for l in [getattr(d, "label")[0] for d in dataset]:
        if l == "positive":
            labels.append(2)
        elif l == "neutral":
            labels.append(1)
        else:
            labels.append(0)

    print(len(labels))
    mean, deviation = svc_rbf(features, labels)
    print("Accuracy mean is %0.2f , and deviation is %0.2f" % (mean,  deviation))


if __name__ == '__main__':
    main()
