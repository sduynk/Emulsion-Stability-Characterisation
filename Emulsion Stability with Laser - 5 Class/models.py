from torchvision.models import (
    resnet50,
    efficientnet_b0,
    convnext_tiny,
    resnet152
)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, num_classes=3, width_scaling=1):
        super(SmallCNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(16 * width_scaling), kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(int(16 * width_scaling)),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(int(16 * width_scaling), int(32 * width_scaling))
        self.layer2 = self._make_layer(int(32 * width_scaling), int(64 * width_scaling))
        self.layer3 = self._make_layer(int(64 * width_scaling), int(128 * width_scaling))
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(128 * width_scaling), num_classes)
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        x = self.stem(x) # 56
        x = self.layer1(x) # 28
        x = self.layer2(x) # 14
        x = self.layer3(x) # 7
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

### functions to get models and optionally load weights

def small_cnn(state_dict_path=None, n_classes=5):
    model = SmallCNN(num_classes=n_classes)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def resnet_18(state_dict_path=None, n_classes=5):
    model = resnet50(weights='DEFAULT')
    # model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def resnet_152(state_dict_path=None, n_classes=5):
    model = resnet152(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def efficientnet(state_dict_path=None, n_classes=5):
    model = efficientnet_b0(weights='DEFAULT')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def convnext(state_dict_path=None, n_classes=5):
    model = convnext_tiny(weights='DEFAULT')
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def init_model(config):
    if config["model"] == "resnet18":
        return resnet_18(n_classes=5)
    elif config["model"] == "resnet152":
        return resnet_152(n_classes=5)
    elif config["model"] == "efficientnet":
        return efficientnet(n_classes=5)
    elif config["model"] == "convnext":
        return convnext(n_classes=5)
    elif config["model"] == "small_cnn":
        return small_cnn(n_classes=5)
    else:
        raise ValueError("Model must be one of 'resnet18', 'resnet152', 'efficientnet', or 'convnext'")
