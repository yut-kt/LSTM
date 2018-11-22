# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from decimal import Decimal, ROUND_HALF_UP

import numpy as np


def main():
    npz = np.load(args.npz_test_file)
    test_labels = [label.argmax() for label in npz['labels']]

    results = []
    for input_file in args.input_files:
        with open(input_file) as p:
            results.append([int(result.strip()) for result in p])

    with open('union_output', mode='w') as p:
        union_results = [int(any(results)) for results in np.transpose(results)]
        p.writelines([f'{result}\n' for result in union_results])

    pre_list = [x == int(y) for x, y in zip(test_labels, union_results) if y == 1]
    rec_list = [x == int(y) for x, y in zip(test_labels, union_results) if x == 1]

    precision = len(list(filter(bool, pre_list))) / len(pre_list) if pre_list else 0
    precision_round = Decimal(str(precision * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    recall = len(list(filter(bool, rec_list))) / len(rec_list) if rec_list else 0
    recall_round = Decimal(str(recall * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    f_measure = 2 * precision * recall / (precision + recall)
    f_measure_round = Decimal(str(f_measure * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    print('---------- Evaluation ----------')
    print(f'{"precision":<9}：{f"{len(list(filter(bool, pre_list)))} / {len(pre_list)}":<9} = {precision_round}%')
    print(f'{"recall":<9}：{f"{len(list(filter(bool, rec_list)))} / {len(rec_list)}":<9} = {recall_round}%')
    print(f'{"F-measure":<9}：{"2pr/(p+r)":<9} = {f_measure_round}%')
    print('--------------------------------')


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-npz', '--npz_test_file', help='テストnpzファイル', required=True)
    parser.add_argument('-i', '--input_files', nargs='+', help='複数入力ファイル')
    args = parser.parse_args()

    main()
