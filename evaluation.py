# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from decimal import Decimal, ROUND_HALF_UP


def main():
    with open(args.answer_file) as a, open(args.input_file) as i:
        a = a.readlines()
        i = i.readlines()
        pre_list = [x == y for x, y in zip(a, i) if y.strip() == '1']
        rec_list = [x == y for x, y in zip(a, i) if x.strip() == '1']

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
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument('-a', '--answer_file', help='答えファイル', required=True)
    parser.add_argument('-i', '--input_file', default='分析結果ファイル', required=True)
    args = parser.parse_args()
    main()
