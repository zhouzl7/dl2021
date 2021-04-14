# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2021/4/13 15:23
@file: conv_visualize.py
@desc: Leverage a proper neural network visualization toolkit to visualize some conv features (10pts)
"""
import torch.nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    model = torch.load('best_model_A.pt')
    # model.cuda()
    model.eval()

    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    count = 0

    def viz(module, input):
        global count
        x = input[0][0].detach().cpu().numpy()
        # Display up to 4 pictures
        min_num = np.minimum(4, x.shape[0])
        for i in range(min_num):
            plt.subplot(1, 4, i + 1)
            plt.imshow(x[i])
        save_path = os.getcwd() + '/convV/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if module.kernel_size[0] == 1:
            plt.savefig(save_path + '{}_downsample.jpg'.format(count))
        else:
            plt.savefig(save_path + '{}.jpg'.format(count))
        count += 1

    for name, m in model.named_modules():
        # show conv features
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)

    img = Image.open('../hw2_dataset/test/AnnualCrop/AnnualCrop_11.jpg')
    img = t(img).unsqueeze(0)
    with torch.no_grad():
        model(img)
