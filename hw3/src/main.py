# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import time


from data import Corpus
from Transformer import TransformerModel
from RNN import RNNModel


parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=35, metavar='N',
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# load data
data_loader = Corpus(train_batch_size=args.train_batch_size,
                     eval_batch_size=args.eval_batch_size,
                     bptt=args.bptt)



# WRITE CODE HERE within two '#' bar
########################################
# bulid your language model here
model = None

########################################


criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)



def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    log_interval = 200

    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        optimizer.zero_grad()

        ########################################
        ######Your code here########
        ########################################



        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data_loader.train_data) // args.bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)

            ########################################
            ######Your code here########
            ########################################





# Train Function
best_val_loss = float("inf")
best_model = None

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, data_loader.val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

########################################

test_loss = evaluate(best_model, data_loader.test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)