import warnings
import nltk
from load_dataset import load_dataset
from models import *
from read_features import read_features

nltk.download("wordnet")
warnings.simplefilter('ignore')


def main():
    # find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/merge_bez2016.csv")
    dataset = load_dataset("~/pycharm/zavrsni/data/main_data.csv")
    # features = [list() for i in range(len(dataset))]
    # boolean_features = extract_boolean_features(dataset)
    # general_features = get_anew([getattr(data, "anot")[1] for data in dataset])

    # for i in range(5):
    #     print("=" * 50)
    #     print(dataset[i].tweet)
    #     print(dataset[i].pos)
    #     print(dataset[i].anot)
    #     print(general_features[i])

    # for i in range(len(dataset)):
    #     features[i].extend(boolean_features[i])
    #     features[i].extend(general_features[i])
    #
    # file = open("/home/ikrizanic/pycharm/zavrsni/data/features.txt", "w")
    # for i in range(len(dataset)):
    #     file.write(str(features[i]))
    #     file.write("\n")
    features = read_features("/home/ikrizanic/pycharm/zavrsni/data/features.txt")

    print(svc_rbf(features, [getattr(d, "label") for d in dataset]))


if __name__ == '__main__':
    main()
