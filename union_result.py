# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import numpy as np


def main():
    npz = np.load(args.npz_test_file)
    test_labels = npz['labels']

    results = []
    for input_file in args.input_files:
        with open(input_file) as p:
            results.append([result.strip() for result in p])

    with open('union_output', mode='w') as p:
        union_results = [int(any(results)) for results in np.transpose(results)]
        p.writelines([f'{result}\n' for result in union_results])

    pre_list = [x == int(y) for x, y in zip(test_labels, results) if y == 1 or y == '1']
    rec_list = [x == int(y) for x, y in zip(test_labels, results) if x == 1 or x == '1']

    precision = len(list(filter(bool, pre_list))) / len(pre_list) * 100 if pre_list else 0
    recall = len(list(filter(bool, rec_list))) / len(rec_list) * 100 if rec_list else 0
    print('precision:', precision)
    print('recall', recall)
    print('---------- ----------')


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-npz', '--npz_test_file', help='テストnpzファイル', required=True)
    parser.add_argument('-i', '--input_files', nargs='+', help='複数入力ファイル')
    args = parser.parse_args()

    main()
