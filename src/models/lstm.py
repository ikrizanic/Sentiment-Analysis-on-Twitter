from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


def compile_model(vocab, embedding_matrix, input_length,
                  trainable=False,
                  recurrent_layer_size=256,
                  dense_size=256,
                  dropout=0.1,
                  recurrent_dropout=0.1,
                  dense_activation='relu',
                  dropout_for_regularization=0.5,
                  output_activation='softmax',
                  optimizer='adam',
                  loss=tf.keras.losses.SquaredHinge(reduction="auto", name="squared_hinge")):
    model = Sequential()
    m = tf.keras.metrics.Recall()
    # Embedding layer
    model.add(
        Embedding(input_dim=len(vocab) + 1,
                  input_length=input_length,
                  output_dim=300,
                  weights=[embedding_matrix],
                  trainable=False,
                  mask_zero=True))

    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(recurrent_layer_size, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout,
                   input_shape=(2048, 28, 300)))
    # model.add(LSTM(recurrent_layer_size, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
    #
    # model.add(LSTM(int(recurrent_layer_size / 2), return_sequences=False))

    # Fully connected layer
    model.add(Dense(dense_size, activation=dense_activation))

    # Dropout for regularization
    model.add(Dropout(dropout_for_regularization))

    # Output layer
    model.add(Dense(3, activation=output_activation))

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.Recall()])
    return model


# CHANGE PATH FOR SERVER
local = "/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/models/model.h5"
djurdja = '/home/ikrizanic/pycharm/zavrsni/models/model.h5'
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(local)]


def fit_model(model, X_train, y_train, X_val, y_val, batch_size=2048, epochs=200):
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val))
    return history, model


def evaluate_model(model, X_test, y_test):
    res = model.evaluate(X_test, y_test)
    return res

def calc_recall(model, test_features, test_labels, path=""):
    import numpy as np
    predictions = model.predict(test_features)
    predictions = np.eye(3, dtype=float)[np.argmax(predictions, axis=1)]
    tp, tn, tu, np, nn, nu = 0, 0, 0, 0, 0, 0
    fp, fn, fu = 0, 0, 0
    for i in range(len(predictions)):
        p = list(predictions[i])
        l = list(test_labels[i])

        if l == [1.0, 0.0, 0.0]:
            nn += 1
            if l == p:
                tn += 1
            else:
                if p == [0.0, 1.0, 0.0]:
                    fu += 1
                else:
                    fp += 1
        if l == [0.0, 1.0, 0.0]:
            nu += 1
            if l == p:
                tu += 1
            else:
                if p == [1.0, 0.0, 0.0]:
                    fn += 1
                else:
                    fp += 1
        if l == [0.0, 0.0, 1.0]:
            np += 1
            if l == p:
                tp += 1
            else:
                if p == [0.0, 1.0, 0.0]:
                    fu += 1
                else:
                    fn += 1
    rp = tp / np
    ru = tu / nu
    rn = tn / nn
    if path != "":
        with open(path, "a") as file:
            file.writelines("True: {}, {}, {}\n".format(tn, tu, tp))
            file.writelines("False: {}, {}, {}\n".format(fn, fu, fp))
            file.writelines("Sum: {}, {}, {}\n".format(nn, nu, np))
            file.writelines("Res: {}, {}, {}\n".format(rn, ru, rp))
    print("True: {}, {}, {}".format(tn, tu, tp))
    print("False: {}, {}, {}".format(fn, fu, fp))
    print("Neto: {}, {}, {}".format(nn, nu, np))
    print("Res: {:4f}, {:4f}, {:4f}".format(rn, ru, rp))
    print("Final: {:4f}".format((rp + rn + ru) * 100 / 3))
    return (rp + rn + ru) * 100 / 3


def calc_recall2(predictions, test_labels, path=""):
    import numpy as np
    test_labels_cat = to_categorical(test_labels)
    tp, tn, tu, np, nn, nu = 0, 0, 0, 0, 0, 0
    fp, fn, fu = 0, 0, 0
    for i in range(len(predictions)):
        p = list(predictions[i])
        l = list(test_labels_cat[i])

        if l == [1.0, 0.0, 0.0]:
            nn += 1
            if l == p:
                tn += 1
            else:
                if p == [0.0, 1.0, 0.0]:
                    fu += 1
                else:
                    fp += 1
        if l == [0.0, 1.0, 0.0]:
            nu += 1
            if l == p:
                tu += 1
            else:
                if p == [1.0, 0.0, 0.0]:
                    fn += 1
                else:
                    fp += 1
        if l == [0.0, 0.0, 1.0]:
            np += 1
            if l == p:
                tp += 1
            else:
                if p == [0.0, 1.0, 0.0]:
                    fu += 1
                else:
                    fn += 1
    rp = tp / np
    ru = tu / nu
    rn = tn / nn
    if path != "":
        with open(path, "a") as file:
            file.writelines("True: {}, {}, {}\n".format(tn, tu, tp))
            file.writelines("False: {}, {}, {}\n".format(fn, fu, fp))
            file.writelines("Sum: {}, {}, {}\n".format(nn, nu, np))
            file.writelines("Res: {}, {}, {}\n".format(rn, ru, rp))
    print("True: {}, {}, {}".format(tn, tu, tp))
    print("False: {}, {}, {}".format(fn, fu, fp))
    print("Neto: {}, {}, {}".format(nn, nu, np))
    print("Res: {:4f}, {:4f}, {:4f}".format(rn, ru, rp))
    print("Final: {:4f}".format((rp + rn + ru) * 100 / 3))
    return (rp + rn + ru) * 100 / 3