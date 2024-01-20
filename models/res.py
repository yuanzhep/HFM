# pre_trained

import os
import torch
import logging
from PIL import Image
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

logging.basicConfig(filename="features_extraction.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock_Baseline(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet34_baseline(pretrained=False):
    model = ResNet_Baseline(BasicBlock_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet34')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def main():
    resnet = models.resnet34(pretrained=True)

    modules = list(resnet.children())[:-3]  # Removed the layer4, avgpool layer and fully connected layer
    resnet = torch.nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1)))  # Added an adaptive avg pooling layer
    torch.save(resnet.state_dict(),
               '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0304_code/resnet_modified_model.pth')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/yz_img/'
    folders = os.listdir(root_dir)

    output_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/0611_yz/output/'
    os.makedirs(output_dir, exist_ok=True)

    for folder in folders:
        logger.info(f"Processing folder: {folder}")
        features_list = []
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)
            for image_name in images:
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(folder_path, image_name)
                    image = Image.open(image_path)
                    input_tensor = preprocess(image)
                    input_batch = input_tensor.unsqueeze(0)

                    resnet.eval()
                    with torch.no_grad():
                        features = resnet(input_batch)
                    features_flat = torch.flatten(features, start_dim=1)
                    features_array = features_flat.squeeze().numpy()
                    features_list.append(features_array)

            features_df = pd.DataFrame(features_list)
            output_csv_path = os.path.join(output_dir, f'{folder}.csv')
            features_df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved features of folder: {folder} to {output_csv_path}")

if __name__ == "__main__":
    main()