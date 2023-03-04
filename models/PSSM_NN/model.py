import torch.nn as nn
import torch

class CNN_BRNN(nn.Module):
    def __init__(self, n_in=21, n_state=3, conv_hidden_size=64, rnn_hidden_size=256):
        super().__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv1 = nn.Sequential(nn.Conv1d( n_in, conv_hidden_size, 3, 1, 3 // 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d( n_in, conv_hidden_size, 7, 1, 7 // 2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d( n_in, conv_hidden_size, 11, 1, 11 // 2), nn.ReLU())

        # nn.GRU(n_input, n_hidden, n_layers, bias, batch_first, dropout, bidirectional)
        # bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        #               Note that this does not apply to hidden or cell states.
        # dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
        # bidirectional – If True, becomes a bidirectional GRU. Default: False

        self.brnn = nn.GRU(conv_hidden_size*3, rnn_hidden_size, 3, True, True, 0.5, True)

        self.fc = nn.Sequential(
                nn.Linear(rnn_hidden_size*2+conv_hidden_size*3, 128),
                nn.ReLU(),
                nn.Linear(128, n_state))

    def forward(self, x):
        # (batch_size , hidden_size , seq_len)
        conv_out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        # (batch_size , seq_len , hidden_size)
        conv_out = conv_out.transpose(1, 2)

        # bidirectional rnn
        out, _ = self.brnn(conv_out)

        out = torch.cat([conv_out, out], dim=2)
        # print(out.sum())

        # Output shape is (batch_size x seq_len x classnum)
        out = self.fc(out)
        #  (N x L x C) --> (N x C x L)
        out = out.transpose(1,2)
        return out
