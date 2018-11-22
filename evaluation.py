# -*- coding: utf-8 -*-


from argparse import ArgumentParser


def main():
    with open(args.answer_file) as a, open(args.input_file) as i:
        pre_list = [x == y for x, y in zip(a, i) if y.strip() == '1']
        rec_list = [x == y for x, y in zip(a, i) if x.strip() == '1']
        print(rec_list)

        precision = len(list(filter(bool, pre_list))) / len(pre_list) if pre_list else 0
        recall = len(list(filter(bool, rec_list))) / len(rec_list) if rec_list else 0

        print(f'精度：{len(list(filter(bool, pre_list)))} / {len(pre_list)} = {precision*100}%')
        print(f'再現率：{len(list(filter(bool, rec_list)))} / {len(rec_list)} = {recall*100}%')
        print(f'F値：{2*precision*recall/(precision+recall)*100}%')
        print('---------- ----------')


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluation')
    parser.add_argument('-a', '--answer_file', help='答えファイル', required=True)
    parser.add_argument('-i', '--input_file', default='分析結果ファイル', required=True)
    args = parser.parse_args()
    main()
