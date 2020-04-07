import warnings
warnings.simplefilter('ignore')
import re
from src.hooks.pretokenization import *
from src.hooks.spell_check import *
from src.load_data.load_dataset import *
from src.features_extraction import *

def main():
    # find_useful_emoticons("~/pycharm/zavrsni/data/emoticons.txt", "~/pycharm/zavrsni/data/merge_bez2016.csv")
    dataset = load_dataset("~/pycharm/zavrsni/data/test.csv")

    boolean_features = extract_boolean_features(dataset)
    print(boolean_features)

    for data in dataset:
        print("="*30)
        print(data.tweet)
        print(data.pos)
        print(data.anot)
        print("\n")


if __name__ == '__main__':
    main()
