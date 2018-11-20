# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from typing import List, Union, Iterator, Any

import MeCab
import numpy as np
from keras.utils.np_utils import to_categorical


def main():
    labels, sentences = get_labels_sentences()
    sentences = np.array(get_word_strList(sentences), dtype=np.str)
    labels = to_categorical(labels, dtype=np.int)

    file_path = f'{args.output_name}.npz'
    np.savez(file_path, sentences=sentences, labels=labels)


def get_labels_sentences() -> Union[Iterator[Any]]:
    """
    ファイルを読み込んでラベルと文に分割
    :return: ラベル配列と文配列のタプル
    """
    with open(args.input_file) as fp:
        return zip(*[
            ('0' if label == '-1' else label, sentence)
            for label, _, sentence in (l.strip().split(maxsplit=2) for l in fp)
        ])


def get_word_strList(sentences: str) -> List[str]:
    """
    文配列から分ち書きした文配列へ変換
    :param sentences: 文配列
    :return: 分ち書きした文配列
    """

    def get_word_str(sentence: str) -> str:
        def validate(word_line: str) -> None | str:
            if word_line.strip() == 'EOS' or word_line == '':
                return
            word, info = word_line.split('\t')
            part, fine_part, _ = info.split(',', maxsplit=2)
            if part == '記号' or fine_part == '数':
                return
            return word

        return ' '.join(
            filter(None,
                   [validate(l) for l in mecab.parse(sentence).split('\n')])
        )

    return [get_word_str(sentence) for sentence in sentences]


if __name__ == '__main__':
    parser = ArgumentParser(description='Make for Keras Dataset')
    parser.add_argument(
        '-i',
        '--input_file',
        help='入力ファイルパス',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_name',
        help='出力ファイル名',
        default='dataset'
    )
    args = parser.parse_args()

    mecab = MeCab.Tagger()

    main()
