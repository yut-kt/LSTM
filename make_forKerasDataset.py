# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from keras.utils.np_utils import to_categorical
import numpy as np
import MeCab


def main():
    labels, sentences = get_tuples_label_sentence()
    sentences = get_word_strList(sentences)

    labels = to_categorical(labels, dtype=np.int)

    file_path = f'{args.output_name}.npz'
    np.savez(file_path, sentences=np.array(sentences, dtype=np.str), labels=labels)


def get_tuples_label_sentence():
    """
    ファイルを読み込んでラベルと文に分割
    :return: ラベル配列と文配列のタプル
    """
    labels, sentences = [], []
    with open(args.input_file) as fp:
        for line in fp:
            label, _, sentence = line.strip().split(maxsplit=2)
            if label == '-1':
                label = '0'

            labels.append(label)
            sentences.append(sentence)
    return labels, sentences


def get_word_strList(sentences):
    """
    文配列から分ち書きした文配列へ変換
    :param sentences: 文配列
    :return: 分ち書きした文配列
    """

    def get_word_str(sentence) -> str:
        def validate(word_line):
            if word_line.strip() == 'EOS' or word_line == '':
                return
            word, info = word_line.split('\t')
            part, fine_part, _ = info.split(',', maxsplit=2)
            if part == '記号' or fine_part == '数':
                return
            return word

        return ' '.join(filter(None, [validate(word_line) for word_line in mecab.parse(sentence).split('\n')]))

    return [get_word_str(sentence) for sentence in sentences]


if __name__ == '__main__':
    parser = ArgumentParser(description='ELMo Feature Based')
    parser.add_argument('-i', '--input_file', help='入力ファイルパス', required=True)
    parser.add_argument('-o', '--output_name', help='出力ファイル名', default='dataset')
    args = parser.parse_args()

    mecab = MeCab.Tagger()

    main()
