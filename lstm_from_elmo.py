# -*- coding: utf-8 -*-

import datetime
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim


def main():
    net = LSTM(1024, 100, True)
    net.cuda()
    print(net)

    dataset = MyDataset(train, trainTag)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8)

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    log_name = datetime.datetime.now().strftime('./log/train-%Y%m%d-%H%M%S.log')
    with open(log_name, mode='w', encoding='utf-8') as wp:
        epoch_loss_list = []
        for epoch in range(500):  # データセットに渡り複数回ループ

            # データ全てのトータルロス
            epoch_loss = 0.0

            for i, data in enumerate(dataloader):
                inputs, labels = data

                temp = []
                for d1 in inputs:
                    temp.append(sum([1 if d2[0] != 0 else 0 for d2 in d1]))
                lens = torch.LongTensor(temp)
                # 大きい順に並び替える
                lens, idx = lens.sort(0, descending=True)
                inputs = inputs[idx]
                labels = labels[idx]

                inputs, labels = Variable(inputs.float().cuda()), Variable(labels.cuda())

                net.batch_size = len(labels)
                optimizer.zero_grad()
                net.hidden = net.init_hidden()

                output = net(inputs, lens.tolist())

                loss = criterion(output, labels)
                epoch_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            epoch_loss_list.append(epoch_loss)
            # ロスの表示
            print("===> Epoch[{}]: Loss: {:.4f}".format(epoch, epoch_loss))
            wp.write("===> Epoch[{}]: Loss: {:.4f}\n".format(epoch, epoch_loss))
        wp.write(str(epoch_loss_list))

    torch.save(net.state_dict(), './data/elmo-results.model')
    torch.save(optimizer.state_dict(), './data/elmo-results.optim')


# DatasetのMyクラス: 入力はword2vecを想定
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tags):
        super(MyDataset, self).__init__()
        assert len(data) == len(tags)
        # npに変換し、0埋めを行う
        max_length = max([len(d) for d in data])
        self.data = np.zeros((len(tags), max_length, len(data[0][0])))
        for i, d1 in enumerate(data):
            for l, d2 in enumerate(d1):
                self.data[i][l] = d2
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        return self.data[index], self.tags[index]


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, bidirectional=False, batch_size=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_dim, bidirectional=bidirectional)
        self.fc0 = nn.Linear(hidden_dim * 2, 40)
        self.fc1 = nn.Linear(40, 2)
        self.hidden = self.init_hidden()

    def forward(self, inputs, lengths):
        # 行と列を入れ替える
        inputs = inputs.transpose(0, 1)
        pack = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        lstm_out, self.hidden = self.lstm(pack, self.hidden)
        # hoge = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        y = self.fc0(torch.cat([self.hidden[0][-1], self.hidden[0][-2]], 1))
        y = self.fc1(F.tanh(y))
        tag_scores = F.log_softmax(y)
        return tag_scores

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # (hidden_state, cell_state)のタプルになる
        num = 2 if self.bidirectional else 1
        return (Variable(torch.zeros(num, self.batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(num, self.batch_size, self.hidden_dim)).cuda())


if __name__ == '__main__':
    NpzFile = np.load('./data/feature.npz')
    train = NpzFile['elmo']
    trainTag = NpzFile['tags']
    main()
