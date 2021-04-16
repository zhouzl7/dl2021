# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2021/4/13 15:17
@file: tSNE_visualize.py
@desc: Visualize the features before the last fully-connected layer using t-SNE (10pts)
"""
import torch
import torch.nn as nn
import data
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def tsne_viz(model, valid_loader):
    model.train(False)

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        out = model(inputs)
        tsne = TSNE(n_components=2)
        y = tsne.fit_transform(out.detach().cpu().numpy())
        fig = plot_embedding(y, labels.numpy(), 't-SNE Visualization: modelA')
        break
    save_path = os.getcwd() + '/result_png/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'tSNE_modelA.png')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # # about model
    num_classes = 10

    # # about data
    data_dir = "../hw2_dataset/"  # # You need to specify the data_dir first
    input_size = 224
    batch_size = 500

    # # model initialization
    model = torch.load('../best_model/best_model_A.pt')
    device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # # data preparation
    _, valid_loader = data.load_data(data_dir=data_dir, test_data_dir="tSNE", input_size=input_size, batch_size=batch_size)
    # # loss function
    criterion = nn.CrossEntropyLoss()
    tsne_viz(model, valid_loader)
