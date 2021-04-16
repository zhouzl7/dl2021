# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2021/4/13 13:58
@file: mainB.py
@desc: 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
from visdom import Visdom


# # Note that: here we provide a basic solution for training and validation.
# # You can directly change it if you find something wrong or not good enough.

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=20):
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    best_acc_t = 0.0
    # # use Visdom to report training and test curves (10pts)
    vis = Visdom()
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        print("LR: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        vis.line(
            X=[epoch],
            Y=[[train_loss, valid_loss]],
            win='loss_modelB_test',
            opts=dict(title='loss_modelB_test', legend=['train_loss', 'valid_loss']),
            update='append')
        vis.line(
            X=[epoch],
            Y=[[train_acc, valid_acc]],
            win='acc_modelB_test',
            opts=dict(title='acc_modelB_test', legend=['train_acc', 'valid_acc']),
            update='append')
        scheduler.step()
        if train_acc > best_acc_t:
            best_acc_t = train_acc
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model_B_test.pt')
    print("best train_acc: {}, best valid_acc: {}".format(best_acc_t, best_acc))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # # about model
    num_classes = 10

    # # about data
    data_dir = "../hw2_dataset/"  # # You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    # # about training
    num_epochs = 200
    steps = [40, 80, 120, 160]
    lr = 0.001

    # # model initialization
    model = models.model_B(num_classes=num_classes)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # # data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, train_data_dir='2-Medium-Scale',
                                                input_size=input_size, batch_size=batch_size)

    # # optimizer
    # data augmentation and learning rate strategy (10pt)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5, amsgrad=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)

    # # loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

