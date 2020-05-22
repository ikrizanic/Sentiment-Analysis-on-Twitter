import pandas as pd

from pretokenization import return_tweets_and_labels


def main():
    local_path = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data"
    djurdja_path = "/home/ikrizanic/pycharm/zavrsni/data"
    working_path = local_path
    data_paths = {"train_dataset": working_path + "/lstm/train_dataset.pl",
                  "test_dataset": working_path + "/lstm/test_dataset.pl",
                  "train_labels": working_path + "/lstm/train_labels.pl",
                  "test_labels": working_path + "/lstm/test_labels.pl",
                  "embedding_matrix": working_path + "/lstm/embedding_matrix.pl"}

    dataset_name = "main" + "_data"
    djurdja_paths = {"dataset": str("~/pycharm/zavrsni/data/" + dataset_name + ".csv"),
                     "labels": "/home/ikrizanic/pycharm/zavrsni/data/labels.txt"}
    local_paths = {
        "dataset": str("/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/" + dataset_name + ".csv"),
        "labels": "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/labels.txt"}

    raw_data_merge = pd.read_csv(local_paths["dataset"], sep="\t", names=["label", "text"])

    test_dataset = pd.read_csv(
        '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/SemEval2017-task4-test.subtask-A.english.txt',
        sep="\t", quotechar='\"', names=["id", "label", "text"])
    # test_dataset = pd.read_csv(
    #     '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/test_data.csv',
    #     sep="\t", quotechar='\"', names=["label", "text"])

    train_dataset, train_labels = return_tweets_and_labels(raw_data_merge)
    test_dataset, test_labels = return_tweets_and_labels(test_dataset)

    for t_main in train_dataset:
        if t_main in test_dataset:
            print(t_main)
            print(test_dataset.index(t_main))


if __name__ == "__main__":
    main()