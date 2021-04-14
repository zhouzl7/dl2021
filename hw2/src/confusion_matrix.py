# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2021/4/13 16:14
@file: confusion_matrix.py
@desc: 
"""
import os
import torch
import data
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(model, valid_loader, device, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    model.train(False)
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        for l, p in zip(labels, predictions):
            matrix[l.item()][p.item()] += 1
    return matrix


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # # about model
    num_classes = 10

    # # about data
    data_dir = "../hw2_dataset/"  # # You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    # # model initialization
    model = torch.load('../best_model/best_model_C.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # # data preparation
    _, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)
    matrix = confusion_matrix(model, valid_loader, device, num_classes)

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = range(0, 10)
    tick_marks = np.arange(len(classes))
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            num = matrix[i, j]
            plt.text(j, i, int(num),
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")

    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    save_path = os.getcwd() + '/result_png/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'confusion_matrix.png')
