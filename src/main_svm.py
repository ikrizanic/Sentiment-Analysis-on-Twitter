from src.feature_extraction.bag_of_words import bag_of_words
from src.feature_extraction.boolean_features import extract_boolean_features
from src.feature_extraction.general_features import get_anew
from src.load_data.load_dataset import *
from src.load_data.pickle_functions import *
from src.hooks.various_functions import *
from src.models.models import *

def main():
    local_path = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data"
    djurdja_path = "/home/ikrizanic/pycharm/zavrsni/data"
    working_path = local_path
    data_paths = {"train_dataset": working_path + "/svm/train_dataset.pl",
                  "input_dataset": working_path + "/svm/input_dataset.pl",
                  "input_labels": working_path + "/svm/input_labels.pl",
                  "test_dataset": working_path + "/svm/test_dataset.pl",
                  "train_labels": working_path + "/svm/train_labels.pl",
                  "test_labels": working_path + "/svm/test_labels.pl",
                  "embedding_matrix": working_path + "/svm/embedding_matrix.pl"}

    dataset_name = "main" + "_data"
    djurdja_paths = {"dataset": str("~/pycharm/zavrsni/data/" + dataset_name + ".csv"),
                     "labels": "/home/ikrizanic/pycharm/zavrsni/data/labels.txt"}
    local_paths = {
        "dataset": str("/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/" + dataset_name + ".csv"),
        "labels": "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/labels.txt"}

    # train_dataset = pd.read_csv(local_paths["dataset"], sep="\t", names=["label", "text"])
    #
    # test_dataset = pd.read_csv(
    #     '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/SemEval2017-task4-test.subtask-A.english.txt',
    #     sep="\t", quotechar='\'', names=["id", "label", "text"])
    #
    # train_dataset, train_labels = return_tweets_and_labels(train_dataset)
    # test_dataset, test_labels = return_tweets_and_labels(test_dataset)
    #
    # train_dataset = process_dataset(train_dataset)
    # test_dataset = process_dataset(test_dataset)
    #
    # dump_dataset(train_dataset, data_paths["train_dataset"])
    # dump_dataset(test_dataset, data_paths["test_dataset"])
    # dump_labels(train_labels, data_paths["train_labels"])
    # dump_labels(test_labels, data_paths["test_labels"])

    print("Reading pickle data...")
    train_dataset = load_dataset(data_paths["train_dataset"])
    test_dataset = load_dataset(data_paths["test_dataset"])
    train_labels = load_labels(data_paths["train_labels"])
    test_labels = load_labels(data_paths["test_labels"])
    print("Done")

    train_features = [list() for i in range(len(train_dataset))]
    test_features = [list() for i in range(len(test_dataset))]

    # train_boolean_features = extract_boolean_features([d['anot'] for d in train_dataset])
    # train_general_features = get_anew([d['anot_tokens'] for d in train_dataset])
    train_vocab, train_features_bow = create_vocab_encode_data([d['anot_tokens'] for d in train_dataset])
    # test_boolean_features = extract_boolean_features([d['anot'] for d in test_dataset])
    # test_general_features = get_anew([d['anot_tokens'] for d in test_dataset])
    test_features_bow = encode_data([d['anot_tokens'] for d in test_dataset], train_vocab)

    for i in range(len(train_features)):
        train_features[i].extend(train_features_bow[i])

    for i in range(len(test_features)):
        test_features[i].extend(test_features_bow[i])

    max_len = max([len(t) for t in train_features])
    train_features = pad_encoded_data(train_features, max_len)
    test_features = pad_encoded_data(test_features, max_len)

    # print("SVC linear cross in progress...")
    # mean, deviation = svc_linear_cross(train_features, train_labels)
    # print("Accuracy mean of svc_linear is %0.2f , and deviation is %0.2f" % (mean,  deviation))
    #
    print("SVC linear in progress...")
    recall = svc_linear(train_features, train_labels, test_features, test_labels)
    print("Recall: {:5.3f}".format(recall))
    # print("Accuracy mean of svc_linear is %0.2f , and deviation is %0.2f" % (mean,  deviation))


if __name__ == '__main__':
    main()
