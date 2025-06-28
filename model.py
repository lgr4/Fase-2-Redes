import torch

class FederatedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 7)
        self.conv2 = torch.nn.Conv2d(20, 40, 7)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(4000, 10)
        self.non_linearity = torch.nn.functional.relu
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'linear': self.linear}

    def forward(self, x):
        x = self.non_linearity(self.conv1(x))
        x = self.non_linearity(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def get_parameters(self): #
        return {name: {'weight': layer.weight.data.clone(), 'bias': layer.bias.data.clone()} for name, layer in self.track_layers.items()} #

    def apply_parameters(self, parameters): #
        with torch.no_grad(): #
            for name in parameters: #
                self.track_layers[name].weight.data.copy_(parameters[name]['weight']) #
                self.track_layers[name].bias.data.copy_(parameters[name]['bias']) #