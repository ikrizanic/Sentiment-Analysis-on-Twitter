import sys
import warnings
import nltk

from src.load_data.read_features import read_features
from src.feature_extraction.boolean_features import extract_boolean_features
from src.feature_extraction.general_features import get_anew
from src.feature_extraction.bag_of_words import bag_of_words
from src.load_data.load_dataset import load_dataset, find_useful_emoticons
from src.models.models import *
from src.models.manual_cross_validation import *


nltk.download("wordnet")
warnings.simplefilter('ignore')


def main(argv):
    if len(argv) > 2:
        dataset_name = argv[2]
    else:
        dataset_name = "main_data"

    find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/main_data.csv")
    dataset = load_dataset(str("~/pycharm/zavrsni/data/" + dataset_name + ".csv"))

    # file = open("/home/ikrizanic/pycharm/zavrsni/data/balanced_features.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(features[i]))
    #     file.write("\n")

    # print("Reading features from file...")
    # features = read_features("/home/ikrizanic/pycharm/zavrsni/data/balanced_features.txt")

    labels = list()
    for lab in [getattr(d, "label")[0] for d in dataset]:
        if lab == "positive":
            labels.append(2)
        elif lab == "neutral":
            labels.append(1)
        else:
            labels.append(0)

    # file = open("/home/ikrizanic/pycharm/zavrsni/data/labels.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(labels[i]))
    #     file.write("\n")

    classifier = SVC(kernel='rbf', random_state=10)
    if argv[1] == "bow":
        features_bow = bag_of_words([getattr(d, "anot")[1] for d in dataset])
        print("Working with BoW...\n")
        scores = cross_validate(classifier, features_bow, labels)
        print(str(scores))
    else:
        print("Working with main features...\n")
        features = [list() for i in range(len(dataset))]
        boolean_features = extract_boolean_features(dataset)
        general_features = get_anew([getattr(data, "anot")[1] for data in dataset])
        #
        for i in range(len(dataset)):
            features[i].extend(boolean_features[i])
            features[i].extend(general_features[i])
        scores = cross_validate(classifier, features, labels)
        print(str(scores))

    # res_file = open("/home/ikrizanic/pycharm/zavrsni/data/res.txt", "w")
    # print("SVC rbf in progress...")
    # mean, deviation = svc_rbf(features, labels)
    # print("Accuracy mean of svc_rbf is %0.2f , and deviation is %0.2f" % (mean,  deviation))
    # res_file.write("RBF\n")
    # res_file.write(str(mean))
    # res_file.write("\n")
    # res_file.write(str(deviation))
    # res_file.write("\n")

    # print("SVC linear in progress...")
    # mean, deviation = svc_linear(features, labels)
    # print("Accuracy mean of svc_linear is %0.2f , and deviation is %0.2f" % (mean,  deviation))
    # res_file.write("Linear\n")
    # res_file.write(str(mean))
    # res_file.write("\n")
    # res_file.write(str(deviation))
    # res_file.write("\n")


if __name__ == '__main__':
    main(sys.argv)
