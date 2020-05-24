import pandas as pd
import tensorflow as tf
import tqdm
import sklearn
from keras.utils import to_categorical
from src.hooks.various_functions import *
from src.load_data.pickle_functions import *
from src.models.lstm import *


# change spellcheck path, and path in lstm (model.h5)
def main():
    local_path = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data"
    djurdja_path = "/home/ikrizanic/pycharm/zavrsni/data"
    working_path = local_path
    data_paths = {"train_dataset": working_path + "/lstm/train_dataset.pl",
                  "input_dataset": working_path + "/lstm/input_dataset.pl",
                  "input_labels": working_path + "/lstm/input_labels.pl",
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

    # train_dataset = pd.read_csv(local_paths["dataset"], sep="\t", names=["label", "text"])
    # train_dataset = pd.read_csv(
    # '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/SemEval2017-task4-dev.subtask-A.english.INPUT.txt',
    #                             sep="\t", quotechar='"',
    #                             names=["id", "label", "text"])

    # test_dataset = pd.read_csv(
    #     '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/SemEval2017-task4-test.subtask-A.english.txt',
    #     sep="\t", quotechar='\'', names=["id", "label", "text"])

    # test_dataset = pd.read_csv(
    #         '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/test_eval.txt',
    #         sep="\t", quotechar='\'', names=["id", "label", "text"])

    # train_dataset, train_labels = return_tweets_and_labels(train_dataset)
    # test_dataset, test_labels = return_tweets_and_labels(test_dataset)

    # train_dataset = process_dataset(train_dataset)
    # test_dataset = process_dataset(test_dataset)

    # dump_dataset(train_dataset, data_paths["train_dataset"])
    # dump_dataset(test_dataset, data_paths["test_dataset"])
    # dump_labels(train_labels, data_paths["train_labels"])
    # dump_labels(test_labels, data_paths["test_labels"])

    # print("Reading pickle data...")
    train_dataset = load_dataset(data_paths["train_dataset"])
    test_dataset = load_dataset(data_paths["test_dataset"])
    train_labels = load_labels(data_paths["train_labels"])
    test_labels = load_labels(data_paths["test_labels"])
    # print("Done")

    print("Train vocab and data encoding...")
    train_vocab, enc_train_data = create_vocab_encode_data([d["anot_tokens"] for d in train_dataset])
    enc_test_data = encode_data([d["anot_tokens"] for d in test_dataset], train_vocab)
    print("Done")

    print("Padding features...")
    train_features = pad_encoded_data(enc_train_data, max(x for x in [len(d) for d in enc_train_data]))
    test_features = pad_encoded_data(enc_test_data, max(x for x in [len(d) for d in enc_train_data]))
    print("Done")

    # print("Embedding matrix...")
    # embedding_matrix = make_embedding_matrix(train_vocab)
    # print("Done")

    # with(open(data_paths["embedding_matrix"], "wb")) as f:
    #     pickle.dump(embedding_matrix, f)
    print("Loading matrix from pickle file...")
    with(open(data_paths["embedding_matrix"], "rb")) as f:
        embedding_matrix = pickle.load(f)
    print("Done")

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1, random_state=25,
                                                      shuffle=True)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    test_labels = to_categorical(test_labels)

    max_len = max(x for x in [len(d) for d in enc_train_data])

    # single run
    cce = tf.keras.losses.CategoricalCrossentropy()
    sq_hinge = tf.keras.losses.SquaredHinge(reduction="auto", name="squared_hinge")
    model = compile_model(train_vocab, embedding_matrix, max_len,
                          recurrent_layer_size=20,
                          dense_size=20,
                          dropout=0,
                          recurrent_dropout=0,
                          dense_activation='relu',
                          dropout_for_regularization=0,
                          output_activation='softmax',
                          optimizer='Adam',
                          loss=cce
                          )

    # history, model = fit_model(model, X_train, y_train, X_val, y_val, batch_size=2048, epochs=5)
    #
    # result = evaluate_model(model, test_features, test_labels)
    # print(result)
    # x_1, x_2, y_1, y_2 = sklearn.model_selection.train_test_split(test_features, test_labels, random_state=13)
    # recall = calc_recall(model, x_1, y_1)
    # recall = calc_recall(model, x_2, y_2)

    # multiple runs
    data = {
        "train_vocab": train_vocab,
        "embedding_matrix": embedding_matrix,
        "max_len": max_len,
        "x_t": X_train,
        "y_t": y_train,
        "x_v": X_val,
        "y_v": y_val,
        "x_test": test_features,
        "y_test": test_labels
    }

    rls = [512]
    dense_size = [512]
    epochs = [100]
    dropout = [0]
    dense_activation = ['relu']
    dropout_for_reg = [0]
    output_activation = ['softmax']
    optimizer = ['adam']
    loss = [tf.keras.losses.CategoricalCrossentropy()]

    params_list = list()
    for r in rls:
        for ds in dense_size:
            for dr in dropout:
                for da in dense_activation:
                    for drop_reg in dropout_for_reg:
                        for oa in output_activation:
                            for opt in optimizer:
                                for ls in loss:
                                    for ep in epochs:
                                        params_list.append({
                                            "recurrent_layer_size": r,
                                            "dense_size": ds,
                                            "dropout": dr,
                                            "dense_activation": da,
                                            "dropout_for_reg": drop_reg,
                                            "output_activation": oa,
                                            "optimizer": opt,
                                            "loss": ls,
                                            "epochs": ep
                                        })

    for params in params_list:
        test_model(data, params, working_path + "/results/24_5_512.txt")


def test_model(data, params, working_path):
    model = compile_model(data["train_vocab"], data["embedding_matrix"], data["max_len"],
                          recurrent_layer_size=params["recurrent_layer_size"],
                          dense_size=params["dense_size"],
                          dropout=params["dropout"],
                          recurrent_dropout=params["dropout"],
                          dense_activation=params["dense_activation"],
                          dropout_for_regularization=params["dropout_for_reg"],
                          output_activation=params["output_activation"],
                          optimizer=params["optimizer"],
                          loss=params["loss"]
                          )
    history, model = fit_model(model, data['x_t'], data['y_t'], data["x_v"], data['y_v'], epochs=params['epochs'])

    result = evaluate_model(model, data['x_test'], data['y_test'])
    print(result)
    recall = calc_recall(model, data['x_test'], data['y_test'], path=working_path)
    print(recall)

    p_out = ""
    for k, v in params.items():
        if k == "loss":
            continue
        p_out += "\n{:20s}{:20s}".format(str(k), str(v))

    # add path
    with open(working_path, "a") as myfile:
        myfile.write("-" * 80)
        myfile.write(p_out)
        myfile.write("Recall:" + str(recall) + "\n")
        myfile.write("-" * 80)
        myfile.write("\n")


if __name__ == '__main__':
    main()
