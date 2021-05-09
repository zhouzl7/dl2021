import os
import torch
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2count = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.idx2count.append(1)
            self.word2idx[word] = len(self.idx2word) - 1
        else:
            self.idx2count[self.word2idx[word]] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, train_batch_size=20, eval_batch_size=10, bptt=35, device='cpu'):
        self.bptt = bptt
        self.device = device
        self.dictionary = Dictionary()
        tokens_train = self.add_corpus(os.path.join(path, 'wiki.train.tokens'))
        tokens_valid = self.add_corpus(os.path.join(path, 'wiki.valid.tokens'))
        tokens_test = self.add_corpus(os.path.join(path, 'wiki.test.tokens'))

        # sort the words by word frequency in descending order
        # this is for using adaptive softmax: it assumes that the most frequent word get index 0
        idx_argsorted = np.flip(np.argsort(self.dictionary.idx2count), axis=-1)

        # re-create given the sorted ones
        self.dictionary.idx2count = np.array(self.dictionary.idx2count)[idx_argsorted].tolist()
        self.dictionary.idx2word = np.array(self.dictionary.idx2word)[idx_argsorted].tolist()
        self.dictionary.word2idx = dict(zip(self.dictionary.idx2word,
                                            np.arange(len(self.dictionary.idx2word)).tolist()))

        train_data = self.tokenize(os.path.join(path, 'wiki.train.tokens'), tokens_train)
        val_data = self.tokenize(os.path.join(path, 'wiki.valid.tokens'), tokens_valid)
        test_data = self.tokenize(os.path.join(path, 'wiki.test.tokens'), tokens_test)

        self.train_data = self.batchify(train_data, train_batch_size)
        self.val_data = self.batchify(val_data, eval_batch_size)
        self.test_data = self.batchify(test_data, eval_batch_size)

    def add_corpus(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        return tokens

    def tokenize(self, path, tokens):
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def get_ntokens(self):
        return len(self.dictionary)
