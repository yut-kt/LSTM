# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, ThresholdedReLU
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.initializers import *


def main():
    npz = np.load(args.npz_train_file)
    train_sentences, labels = npz['sentences'], npz['labels']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    sequences = tokenizer.texts_to_sequences(train_sentences)
    max_length = max([len(sentence) for sentence in train_sentences])
    train = pad_sequences(sequences, max_length).reshape(len(train_sentences), 1, max_length)

    model = KerasClassifier(build_fn=create_model, epochs=1, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=create_param_grid(), cv=3, n_jobs=-1)
    grid = grid.fit(train, labels)

    print(grid.best_params_)
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def create_model(kernel_initializer1=glorot_uniform(),
                 kernel_initializer2='glorot_uniform',
                 kernel_initializer3='glorot_uniform',
                 kernel_initializer4='glorot_uniform',
                 kernel_initializer5='glorot_uniform',
                 kernel_initializer6='glorot_uniform',
                 kernel_initializer7='glorot_uniform',
                 kernel_initializer8='glorot_uniform',

                 recurrent_activation1='hard_sigmoid',
                 recurrent_activation2='hard_sigmoid',
                 recurrent_activation3='hard_sigmoid',
                 recurrent_activation4='hard_sigmoid',

                 optimizer='RMSprop',
                 lr=0.001,
                 rho=0.9):
    input_size = 644
    model = Sequential()
    model.add(
        LSTM(200,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation=recurrent_activation1,
             kernel_initializer=kernel_initializer1,
             return_sequences=True))
    model.add(
        LSTM(100,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation=recurrent_activation2,
             return_sequences=True))
    model.add(
        LSTM(80,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation=recurrent_activation3,
             return_sequences=True))
    model.add(
        LSTM(60,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation=recurrent_activation4,
             return_sequences=False))
    model.add(Dense(60))
    model.add(Activation('softplus'))
    model.add(Dense(40))
    model.add(ThresholdedReLU())
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('hard_sigmoid'))

    if optimizer == 'RMSProp':
        optimizer = RMSprop(lr=lr, rho=rho)

    model.compile(loss='cosine_proximity', optimizer=optimizer, metrics=['acc'])
    return model


def create_param_grid():
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # lr = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # rho = [0.1, 0.5, 0.9, 1.2, 1.5]
    kernel_initializer = [Zeros(), Ones(), Constant(), RandomNormal(), RandomUniform(), TruncatedNormal(),
                          VarianceScaling(), Orthogonal(), Identity(), glorot_normal(), glorot_uniform(), he_normal(),
                          lecun_normal(), he_uniform(), lecun_uniform(), ]

    return dict(
        kernel_initializer1=kernel_initializer
    )


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-t', '--npz_train_file', help='学習npzファイル', required=True)
    args = parser.parse_args()
    main()
