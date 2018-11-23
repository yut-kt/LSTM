# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, ThresholdedReLU
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def main():
    npz = np.load(args.npz_train_file)
    train_sentences, labels = npz['sentences'], npz['labels']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    sequences = tokenizer.texts_to_sequences(train_sentences)
    max_length = max([len(sentence) for sentence in train_sentences])
    train = pad_sequences(sequences, max_length).reshape(len(train_sentences), 1, max_length)

    model = KerasClassifier(build_fn=create_model(train.shape[2]))
    grid = GridSearchCV(estimator=model, param_grid=create_param_grid())
    grid_result = grid.fit(train, labels)

    print(grid_result.best_params_)
    print(grid_result.cv_results_)


def create_model(input_size):
    model = Sequential()
    model.add(
        LSTM(200,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(100,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(80,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(60,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=False))
    model.add(Dense(60))
    model.add(Activation('softplus'))
    model.add(Dense(40))
    model.add(ThresholdedReLU())
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('hard_sigmoid'))
    model.compile(loss='cosine_proximity', optimizer=RMSprop(), metrics=['acc'])
    return model


def create_param_grid():
    epochs = [1, 3, 5]
    batch_size = [100, 200, 300]
    return dict(
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-npz', '--npz_train_file', help='学習npzファイル', required=True)
    args = parser.parse_args()
    main()
