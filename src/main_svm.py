import sys
import warnings
import nltk
from src.load_data.read_features import read_features
from src.feature_extraction.boolean_features import extract_boolean_features
from src.feature_extraction.general_features import get_anew
from src.feature_extraction.bag_of_words import bag_of_words
from src.load_data.load_dataset import *
from src.models.models import *
from src.models.manual_cross_validation import *


def main():
    dataset_name = "main_dataset"
    # find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/main_data.csv")
    dataset = load_dataset(str("~/pycharm/zavrsni/data/" + dataset_name + ".csv"))

    # print("Reading features from file...")
    # features = read_features("/home/ikrizanic/pycharm/zavrsni/data/balanced_features.txt")

    labels = make_labels(dataset)

    print_data(dataset)

    # file = open("/home/ikrizanic/pycharm/zavrsni/data/labels.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(labels[i]))
    #     file.write("\n")

    # classifier = SVC(kernel='rbf', random_state=10)
    # if argv[1] == "bow":
    #     features_bow = bag_of_words([getattr(d, "anot")[1] for d in dataset])
    #     print("Working with BoW...\n")
    #     scores = cross_validate(classifier, features_bow, labels)
    #     print(str(scores))
    # else:
    #     print("Working with main features...\n")

    features = [list() for i in range(len(dataset))]
    # boolean_features = extract_boolean_features(dataset)
    general_features = get_anew([getattr(data, "anot")[1] for data in dataset])
    # features_bow = bag_of_words([getattr(d, "anot")[1] for d in dataset])

    #
    for i in range(len(dataset)):
        # features[i].extend(boolean_features[i])
        features[i].extend(general_features[i])
    for line in features:
        print(line)
    #
    # file = open("/home/ikrizanic/pycharm/zavrsni/data/balanced_features.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(features[i]))
    #     file.write("\n")
    #
    # file = open("/home/ikrizanic/pycharm/zavrsni/data/bow_features.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(features_bow[i]))
    #     file.write("\n")
    #
    # for i in range(len(dataset)):
    #     features[i].extend(features_bow[i])
    #
    print("SVC rbf in progress...")
    mean, deviation = svc_rbf(features, labels)
    print("Accuracy mean of svc_rbf is %0.2f , and deviation is %0.2f" % (mean,  deviation))
    #
    # print("SVC linear in progress...")
    # mean, deviation = svc_linear(features, labels)
    # print("Accuracy mean of svc_linear is %0.2f , and deviation is %0.2f" % (mean,  deviation))


if __name__ == '__main__':
    main()