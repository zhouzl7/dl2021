# coding: utf-8
import argparse
import math
import os

import torch
import torch.nn as nn
import time
from visdom import Visdom

from data import Corpus
from Transformer import TransformerModel
from RNN import RNNModel

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--model', type=str, default='RNN',
                    help='type of recurrent net (RNN, Transformer)')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--ninput', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=100, metavar='N',
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Use gpu or cpu to train
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")
print("device:{}".format(device))

print('using Visdom for visualize')
viz = Visdom()

# load data
data_loader = Corpus(train_batch_size=args.train_batch_size,
                     eval_batch_size=args.eval_batch_size,
                     bptt=args.bptt)

# WRITE CODE HERE within two '#' bar
########################################
# bulid your language model here
nvoc = data_loader.get_ntokens()
nhead = 0  # todo
steps = [40, 100]
if args.model == 'RNN':
    model = RNNModel(args.rnn_type, nvoc, args.ninput, args.nhid, args.nlayers).to(device)
elif args.model == 'Transformer':
    model = TransformerModel(nvoc, args.ninput, nhead, args.nhid, args.nlayers).to(device)
########################################

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
# lr = 5.0  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train():
    model.train()  # Turn on the train mode
    ret_loss = 0.
    log_count = 0
    total_loss = 0.
    start_time = time.time()
    forward_elapsed_time = 0.
    log_interval = 200
    hidden = model.init_hidden(args.train_batch_size)

    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)

        # synchronize cuda for a proper speed benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_start_time = time.time()

        optimizer.zero_grad()

        ########################################
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, nvoc), targets)
        total_loss += loss.item()

        # synchronize cuda for a proper speed benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        ########################################

        if batch % log_interval == 0 and batch > 0:
            log_count += 1
            cur_loss = total_loss / log_interval
            ret_loss += cur_loss
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | forward ms/batch {:5.2f} | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_loader.train_data) // args.bptt, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval, forward_elapsed_time * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0.
            start_time = time.time()
            forward_elapsed_time = 0.
    return ret_loss / log_count


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)

            ########################################
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, nvoc), targets)
            total_loss += len(data) * loss.item()
            hidden = repackage_hidden(hidden)
            ########################################
    loss_mean = total_loss / len(data_source)
    print('[eval] loss: {}, LR: {}'.format(loss_mean, optimizer.state_dict()['param_groups'][0]['lr']))
    return loss_mean


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
    # Train Function
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        val_loss = evaluate(model, data_loader.val_data)
        viz.line(
            X=[epoch],
            Y=[[train_loss, val_loss]],
            win=args.model+'_loss' if args.model == 'Transformer' else args.rnn_type + '_loss',
            opts=dict(title=args.model+'_loss' if args.model == 'Transformer' else args.rnn_type + '_loss',
                      legend=['train_loss', 'valid_loss']),
            update='append')
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
              'train ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         train_loss, math.exp(train_loss)))
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(best_model, args.model+'_best_model.pt')

        scheduler.step()

    ########################################

    test_loss = evaluate(best_model, data_loader.test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
