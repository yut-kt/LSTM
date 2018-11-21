# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def main():
    with open(args.tokenizer, 'rb') as p:
        tokenizer = pickle.load(p)

    model = load_model(args.model, compile=False)
    max_length = model.get_config()['layers'][0]['config']['batch_input_shape'][2]

    npz = np.load(args.npz_test_file)
    test_sentences = npz['sentences']
    test_labels = npz['labels']

    tests_length = len(test_sentences)
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test = pad_sequences(test_sequences, max_length, padding='pre').reshape(tests_length, 1, max_length)
    test_labels = [label.argmax() for label in test_labels]

    model = load_model(args.model)

    results = [result.argmax() for result in model.predict(test)]

    with open('output_lstm.txt', mode='w') as p:
        for result in results:
            p.write(f'{result}\n')

    f_list = [x == int(y) for x, y in zip(test_labels, results)]
    pre_list = [x == int(y) for x, y in zip(test_labels, results) if y == 1 or y == '1']
    rec_list = [x == int(y) for x, y in zip(test_labels, results) if x == 1 or x == '1']

    f = len(list(filter(bool, f_list))) / len(f_list) * 100
    precision = len(list(filter(bool, pre_list))) / len(pre_list) * 100 if pre_list else 0
    recall = len(list(filter(bool, rec_list))) / len(rec_list) * 100 if rec_list else 0
    print('f:', f)
    print('precision:', precision)
    print('recall', recall)
    print('---------- ----------')


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-npz', '--npz_test_file', help='テストnpzファイル', required=True)
    parser.add_argument('-m', '--model', help='学習したモデル', required=True)
    parser.add_argument('-t', '--tokenizer', help='Tokenizer pkl', required=True)
    args = parser.parse_args()

    main()
