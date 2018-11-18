# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np
from statistics import mean

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer


def main():
    npz = np.load(args.npz_train_file)
    train_sentences, labels = npz['sentences'], npz['labels']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    sequences = tokenizer.texts_to_sequences(train_sentences)

    sentence_lengths = [len(sentence) for sentence in train_sentences]

    max_length = max(sentence_lengths)
    mean_length = int(mean(sentence_lengths))
    middle_length = int((max_length + mean_length) // 2)

    sentences_length = len(train_sentences)

    train1 = pad_sequences(sequences, max_length, padding='pre').reshape(sentences_length, 1, max_length)
    train2 = pad_sequences(sequences, mean_length, padding='pre').reshape(sentences_length, 1, mean_length)
    train3 = pad_sequences(sequences, middle_length, padding='pre').reshape(sentences_length, 1, middle_length)
    train4 = pad_sequences(sequences, max_length, padding='post').reshape(sentences_length, 1, max_length)
    train5 = pad_sequences(sequences, mean_length, padding='post').reshape(sentences_length, 1, mean_length)
    train6 = pad_sequences(sequences, middle_length, padding='post').reshape(sentences_length, 1, middle_length)

    npz = np.load(args.npz_test_file)
    test_sentences = npz['sentences']
    test_labels = npz['labels']

    tests_length = len(test_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test1 = pad_sequences(test_sequences, max_length, padding='pre').reshape(tests_length, 1, max_length)
    test2 = pad_sequences(test_sequences, mean_length, padding='pre').reshape(tests_length, 1, mean_length)
    test3 = pad_sequences(test_sequences, middle_length, padding='pre').reshape(tests_length, 1, middle_length)
    test4 = pad_sequences(test_sequences, max_length, padding='post').reshape(tests_length, 1, max_length)
    test5 = pad_sequences(test_sequences, mean_length, padding='post').reshape(tests_length, 1, mean_length)
    test6 = pad_sequences(test_sequences, middle_length, padding='post').reshape(tests_length, 1, middle_length)

    epochs = 1
    model = create_model(train1.shape[2])
    model.fit(train1, labels, epochs=epochs)
    result1 = model.predict(test1)

    model = create_model(train2.shape[2])
    model.fit(train2, labels, epochs=epochs)
    result2 = model.predict(test2)

    model = create_model(train3.shape[2])
    model.fit(train3, labels, epochs=epochs)
    result3 = model.predict(test3)

    model = create_model(train4.shape[2])
    model.fit(train4, labels, epochs=epochs)
    result4 = model.predict(test4)

    model = create_model(train5.shape[2])
    model.fit(train5, labels, epochs=epochs)
    result5 = model.predict(test5)

    model = create_model(train6.shape[2])
    model.fit(train6, labels, epochs=epochs)
    result6 = model.predict(test6)

    results = (result1 + result2 + result3 + result4 + result5 + result6) / 6
    results = [result.argmax() for result in results]

    print(results)

    test_labels = [label.argmax() for label in test_labels]
    f_list = [x == int(y) for x, y in zip(test_labels, results)]
    pre_list = [x == int(y) for x, y in zip(test_labels, results) if y == 1 or y == '1']
    rec_list = [x == int(y) for x, y in zip(test_labels, results) if x == 1 or x == '1']

    f = len(list(filter(bool, f_list))) / len(f_list) * 100
    precision = len(list(filter(bool, pre_list))) / len(pre_list) * 100 if pre_list else 0
    recall = len(list(filter(bool, rec_list))) / len(rec_list) * 100 if rec_list else 0
    print('f:', f)
    print('precision:', precision)
    print('recall', recall)


def create_model(input_size):
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(None, 1, input_size), return_sequences=False))
    model.add(Dense(80))
    model.add(Dense(40))
    model.add(Dense(2))
    model.add(Activation("relu"))
    # optimizer = Adam(lr=0.001)
    optimizer = Adam(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
    return model


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-train', '--npz_train_file', help='学習npzファイル', required=True)
    parser.add_argument('-test', '--npz_test_file', help='テストnpzファイル', required=True)
    args = parser.parse_args()

    main()
