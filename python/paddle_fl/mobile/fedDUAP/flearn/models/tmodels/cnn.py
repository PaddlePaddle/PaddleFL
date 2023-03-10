"""
CNN
"""
import torch
from torch import nn
import collections

class CNNOrigin(nn.Module):
    """
    CNN origin
    """
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
        super(CNNOrigin, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [32, 'M', 64, 'M', 64]
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * self.config[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

        self.loss_func = nn.CrossEntropyLoss()


    def _make_feature_layers(self):
        """
        make feature layers
        """
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=0),
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def generate_from_pd_model(self, state_pd):
        """
        generate from pd model
        """
        state_pd_to_pt = collections.OrderedDict()
        for k, v in state_pd.items():
            state_pd_to_pt[k] = torch.tensor(v.clone().numpy())
            if 'classifier' in k and 'weight' in k:
                state_pd_to_pt[k] = state_pd_to_pt[k].T
        self.load_state_dict(state_pd_to_pt)

if __name__ == "__main__":
    from flearn.models.cnn import CNN as CNN_pd
    import collections
    model_pd = CNN_pd()
    model_pt = CNNOrigin()

    state_pd = model_pd.state_dict()

    state_pd_to_pt = collections.OrderedDict()
    for k, v in state_pd.items():
        state_pd_to_pt[k] = torch.tensor(v.clone().numpy())
        if 'classifier' in k and 'weight' in k:
            state_pd_to_pt[k] = state_pd_to_pt[k].T
    state_pt = model_pt.state_dict()
    model_pt.load_state_dict(state_pd_to_pt)