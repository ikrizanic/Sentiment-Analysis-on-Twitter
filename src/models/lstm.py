from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from scipy.constants import hp


def compile_model(vocab, embedding_matrix, input_length,
                  trainable=False,
                  recurrent_layer_size=128,
                  dense_size=128,
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
    model.add(LSTM(recurrent_layer_size, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout))

    # Second layer
    model.add(LSTM(recurrent_layer_size, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout))

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
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(local)]


def fit_model(model, X_train, y_train, X_val, y_val, batch_size=2048):
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, epochs=150,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val))
    return history, model


def evaluate_model(model, X_test, y_test):
    res = model.evaluate(X_test, y_test)
    return res
