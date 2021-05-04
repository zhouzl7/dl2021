import math
import torch
import torch.nn as nn

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        ########################################
        ######Your code here########
        ########################################

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        ########################################
        ######Your code here########
        ########################################
        pass



    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        ########################################
        ######Your code here########
        ########################################
        pass


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        ########################################
        ######Your code here########
        ########################################
        pass

    def forward(self, x):
        ########################################
        ######Your code here########
        ########################################
        pass