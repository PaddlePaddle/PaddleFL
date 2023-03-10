"""
lenet
"""
import torch
import torch.nn as nn
import collections

class LENETOrigin(nn.Module):
    """
    LeNet Origin
    """
    def __init__(self, in_channel=3, num_classes=10, config=None):
        super(LENETOrigin, self).__init__()
        if config is None:
            config = [6, 16]
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, config[0], 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(config[0], config[1], 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(config[1] * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
            nn.ReLU(inplace=True),
        )

        self.loss_func = nn.CrossEntropyLoss()

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

if __name__  ==  "__main__":
    from flearn.models.lenet import LENET as LENET_pd
    import collections
    model_pd = LENET_pd()
    model_pt = LENETOrigin()

    state_pd = model_pd.state_dict()

    state_pd_to_pt = collections.OrderedDict()
    for k, v in state_pd.items():
        state_pd_to_pt[k] = torch.tensor(v.clone().numpy())
        if 'classifier' in k and 'weight' in k:
            state_pd_to_pt[k] = state_pd_to_pt[k].T
    state_pt = model_pt.state_dict()
    model_pt.load_state_dict(state_pd_to_pt)