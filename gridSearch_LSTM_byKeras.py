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


def main():
    npz = np.load(args.npz_train_file)
    train_sentences, labels = npz['sentences'], npz['labels']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    sequences = tokenizer.texts_to_sequences(train_sentences)
    max_length = max([len(sentence) for sentence in train_sentences])
    train = pad_sequences(sequences, max_length).reshape(len(train_sentences), 1, max_length)

    model = KerasClassifier(build_fn=create_model, epochs=3, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=create_param_grid(), cv=3, n_jobs=-1)
    grid = grid.fit(train, labels)

    print(grid.best_params_)
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    npz = np.load(args.npz_test_file)
    test_sentences = npz['sentences']
    tests_length = len(test_sentences)
    test_labels = npz['labels']
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test = pad_sequences(test_sequences, max_length, padding='pre').reshape(tests_length, 1, max_length)
    test_labels = [label.argmax() for label in test_labels]

    results = [result.argmax() for result in grid.predict(test)]

    pre_list = [x == int(y) for x, y in zip(test_labels, results) if y == 1 or y == '1']
    rec_list = [x == int(y) for x, y in zip(test_labels, results) if x == 1 or x == '1']

    precision = len(list(filter(bool, pre_list))) / len(pre_list) if pre_list else 0
    precision_round = Decimal(str(precision * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    recall = len(list(filter(bool, rec_list))) / len(rec_list) if rec_list else 0
    recall_round = Decimal(str(recall * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    f_measure = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    f_measure_round = Decimal(str(f_measure * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    print('---------- Evaluation ----------')
    print(f'{"precision":<9}：{f"{len(list(filter(bool, pre_list)))} / {len(pre_list)}":<9} = {precision_round}%')
    print(f'{"recall":<9}：{f"{len(list(filter(bool, rec_list)))} / {len(rec_list)}":<9} = {recall_round}%')
    print(f'{"F-measure":<9}：{"2pr/(p+r)":<9} = {f_measure_round}%')
    print('--------------------------------')


def create_model(recurrent_activation1='hard_sigmoid',
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
    lr = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    rho = [0.1, 0.5, 0.9, 1.2, 1.5]
    return dict(
        lr=lr,
        rho=rho,
    )


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-train', '--npz_train_file', help='学習npzファイル', required=True)
    parser.add_argument('-test', '--npz_test_file', help='テストnpzファイル', required=True)
    args = parser.parse_args()
    main()
