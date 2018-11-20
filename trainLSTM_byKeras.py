# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, ThresholdedReLU
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer


def main():
    npz = np.load(args.npz_train_file)
    train_sentences, labels = npz['sentences'], npz['labels']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    with open(f'{args.tokenizer_name}.json', mode='w') as p:
        p.write(tokenizer.to_json(indent=2))

    sequences = tokenizer.texts_to_sequences(train_sentences)
    max_length = max([len(sentence) for sentence in train_sentences])
    train = pad_sequences(sequences, max_length).reshape(len(train_sentences), 1, max_length)

    model = create_model(train.shape[2])
    model.fit(train, labels, epochs=3)
    model.save(f'{args.model_name}.h5', include_optimizer=False)


def create_model(input_size):
    model = Sequential()
    model.add(
        LSTM(200,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(100,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(80,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=True))
    model.add(
        LSTM(60,
             batch_input_shape=(None, 1, input_size),
             activation='softsign',
             recurrent_activation='hard_sigmoid',
             return_sequences=False))
    model.add(Dense(60))
    model.add(Activation('softplus'))
    model.add(Dense(40))
    model.add(ThresholdedReLU())
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('hard_sigmoid'))
    model.compile(loss='cosine_proximity', optimizer=RMSprop(), metrics=['acc'])
    return model


if __name__ == '__main__':
    parser = ArgumentParser(description='LSTM by Keras')
    parser.add_argument('-tr', '--npz_train_file', help='学習npzファイル', required=True)
    parser.add_argument('-m', '--model_name', help='モデルの保存名', default='model')
    parser.add_argument('-to', '--tokenizer_name', help='Tokenizerの保存名', default='tokenizer')
    args = parser.parse_args()
    main()
