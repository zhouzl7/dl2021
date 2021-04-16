from torchvision import models
import torch.nn as nn
from DilateNet import dilateNet18, dilateNet34


def model_A(num_classes):
    model_resnet = models.resnet50(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_B(num_classes):
    # # your code here
    model = dilateNet18(num_classes)

    # extra techniques(20pt): Kaiming Normal for weight initialization
    def initialize_weights(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    if len(m.state_dict()[key].shape) != 1:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    else:
                        m.state_dict()[key][...] = 1
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    model.apply(initialize_weights)
    return model


def model_C(num_classes):
    # # your code here
    model = dilateNet18(num_classes)

    # extra techniques(20pt): Kaiming Normal for weight initialization
    # def initialize_weights(m):
    #     for key in m.state_dict():
    #         if key.split('.')[-1] == 'weight':
    #             if 'conv' in key:
    #                 if len(m.state_dict()[key].shape) != 1:
    #                     nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
    #                 else:
    #                     m.state_dict()[key][...] = 1
    #             if 'bn' in key:
    #                 m.state_dict()[key][...] = 1
    #         elif key.split('.')[-1] == 'bias':
    #             m.state_dict()[key][...] = 0
    #
    # model.apply(initialize_weights)
    return model
