import pandas as pd
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
                  "test_dataset": working_path + "/lstm/test_dataset.pl",
                  "train_labels": working_path + "/lstm/train_labels.pl",
                  "test_labels": working_path + "/lstm/test_labels.pl",
                  "embedding_matrix": working_path + "/lstm/embedding_matrix.pl"}

    # dataset_name = "main" + "_data"
    # djurdja_paths = {"dataset": str("~/pycharm/zavrsni/data/" + dataset_name + ".csv"),
    #                  "labels": "/home/ikrizanic/pycharm/zavrsni/data/labels.txt"}
    # local_paths = {
    #     "dataset": str("/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/" + dataset_name + ".csv"),
    #     "labels": "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/labels.txt"}
    #
    # raw_data_merge = pd.read_csv(local_paths["dataset"], sep="\t", names=["label", "text"])
    #
    # test_dataset = pd.read_csv(
    #     '/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/SemEval2017-task4-test.subtask-A.english.txt',
    #     sep="\t", quotechar='\'', names=["id", "label", "text"])
    #
    # train_dataset, train_labels = return_tweets_and_labels(raw_data_merge)
    # test_dataset, test_labels = return_tweets_and_labels(test_dataset)
    #
    # train_dataset = process_dataset(train_dataset)
    # test_dataset = process_dataset(test_dataset)

    # dump_dataset(train_dataset, )
    # dump_dataset(test_dataset, )
    # dump_labels(train_labels, )
    # dump_labels(test_labels, )

    print("Reading pickle data...")
    train_dataset = load_dataset(data_paths["train_dataset"])
    test_dataset = load_dataset(data_paths["test_dataset"])
    train_labels = load_labels(data_paths["train_labels"])
    test_labels = load_labels(data_paths["test_labels"])

    print("Train vocab and data encoding...")
    train_vocab, enc_train_data = create_vocab_encode_data([d["anot_tokens"] for d in train_dataset])
    enc_test_data = encode_data(test_dataset, train_vocab)

    print("Padding features...")
    train_features = pad_encoded_data(enc_train_data, max(x for x in [len(d) for d in enc_train_data]))
    test_features = pad_encoded_data(enc_test_data, max(x for x in [len(d) for d in enc_train_data]))

    # embedding_matrix = make_embedding_matrix(train_vocab)

    # with(open(data_paths["embedding_matrix"], "wb")) as f:
    #     pickle.dump(embedding_matrix, f)

    print("Loading matrix from pickle file...")
    with(open(data_paths["embedding_matrix"], "rb")) as f:
        embedding_matrix = pickle.load(f)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.4, random_state=13)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    test_labels = to_categorical(test_labels)

    max_len = max(x for x in [len(d) for d in enc_train_data])
    model = compile_model(train_vocab, embedding_matrix, max_len)
    history, model = fit_model(model, X_train, y_train, X_val, y_val)
    for i in range(5):
        result = evaluate_model(model, test_features, test_labels)
        print("-" * 80)
        print("Loss: " + str(result[0]))
        print("Acc: " + str(result[1]))

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

    rls = [128, 256]
    dense_size = [64, 128, 256]
    dropout = [0, 0.05, 0.1, 0.2, 0.4]
    dense_activation = ['relu']
    dropout_for_reg = [0, 0.5]
    output_activation = ['softmax']
    optimizer = ['adam']
    loss = [tf.keras.losses.SquaredHinge(reduction="auto", name="squared_hinge")]

    params_list = list()
    for r in rls:
        for ds in dense_size:
            for dr in dropout:
                for da in dense_activation:
                    for drop_reg in dropout_for_reg:
                        for oa in output_activation:
                            for opt in optimizer:
                                for ls in loss:
                                    params_list.append({
                                        "recurrent_layer_size": r,
                                        "dense_size": ds,
                                        "dropout": dr,
                                        "dense_activation": da,
                                        "dropout_for_reg": drop_reg,
                                        "output_activation": oa,
                                        "optimizer": opt,
                                        "loss": ls
                                    })

    # for params in params_list:
    #     test_model(data, params)


def test_model(data, params):
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
    history, model = fit_model(model, data['x_t'], data['y_t'], data["x_v"], data['y_v'])

    # TODO: dodati višestruku evaluaciju i računati avg
    result = evaluate_model(model, data['x_test'], data['y_test'])
    print(params)
    print("Loss: " + str(result[0]))
    print("Acc: " + str(result[1]))

    p_out = ""
    for k, v in params.items():
        if k == "loss":
            continue
        p_out += "\n{:20s}{:20s}".format(str(k), str(v))

    with open("result_15_5.txt", "a") as myfile:
        myfile.write("-" * 80)
        myfile.write(p_out)
        myfile.write("Loss: " + str(result[0]) + "\n")
        myfile.write("Acc: " + str(result[1]) + "\n")
        myfile.write("-" * 80)
        myfile.write("\n")


if __name__ == '__main__':
    main()
