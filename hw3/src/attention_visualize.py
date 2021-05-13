# coding: utf-8
import argparse
import os

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data import Corpus
plt.rcParams.update({'figure.max_open_warning': 0})

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--data', type=str, default='./.data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=100, metavar='N',
                    help='sequence length')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Use gpu or cpu to train
if args.cuda:
    device = torch.device("cuda:" + str(args.gpu_id))
else:
    device = torch.device("cpu")
print("device:{}".format(device))

# load data
data_loader = Corpus(path=args.data, train_batch_size=args.train_batch_size,
                     eval_batch_size=args.eval_batch_size,
                     bptt=args.bptt, device=device)


def show_attention(word, input_sentence, attentions, index=80):
    # 用color bar设置图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions[0][0][index-20:index+5, index:index+1].cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # 设置坐标
    ax.set_xticklabels([''] + word.split(' ') +
                       ['<EOS>'])  # , rotation=90
    ax.set_yticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'])

    # 在每个刻度处显示标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    save_path = os.getcwd() + '/attn_viz/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'attn' + str(index) + '.jpg')


def attention_viz(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    src_mask = eval_model.generate_square_subsequent_mask(args.bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)

            ########################################
            if data.size(0) != args.bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output, attention = eval_model(data, src_mask)
            plt.figure()
            plt.matshow(attention[0][0].cpu().numpy(), cmap='bone')
            save_path = os.getcwd() + '/attn_viz/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + 'attn.jpg')
            temp = data.permute(1, 0)[0]
            sentences = ""
            word = None
            idx = 80
            for j in range(25):
                j += idx-20
                if j == idx:
                    word = data_loader.dictionary.idx2word[temp[j]]
                sentences += data_loader.dictionary.idx2word[temp[j]] + " "
            print(word)
            print(sentences)
            show_attention(word, sentences, attention, index=idx)
            break


if __name__ == "__main__":
    best_model = torch.load('./Transformer24_best_model.pt', map_location=device)
    if torch.cuda.is_available():
        best_model.cuda()
    best_model.eval()

    attention_viz(best_model, data_loader.test_data)
