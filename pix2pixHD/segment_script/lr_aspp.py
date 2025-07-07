import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from modelio import store_config_args, LoadableModel


class LiteRASPP(LoadableModel):
    @store_config_args
    def __init__(self, backbone: str = 'resnet18', n_class: int = 1, use_64_low_level_features: bool = False,
                 spatial_dim: int = 128):
        super(LiteRASPP, self).__init__()
        backbone_models = {'resnet18': models.resnet18}  # pretraining on ImageNet
        self.spatial_dim = spatial_dim

        self.low_channel = 64 if use_64_low_level_features else 128
        self.inter_channel = 128  # channel number of LR-ASPP head
        self.high_channel = 256

        layer_low_feature = 'relu' if use_64_low_level_features else 'layer2'
        self.encoder = models._utils.IntermediateLayerGetter(backbone_models[backbone](pretrained=True),
                                                             {layer_low_feature: 'low_channel',
                                                              'layer3': 'high_channel'})
        self.squeeze_and_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.high_channel, out_channels=self.inter_channel, kernel_size=1, bias=False),
            # no batchnorm and bias like in the paper
            nn.Sigmoid()
        )

        self.CBR = nn.Sequential(
            nn.Conv2d(self.high_channel, self.inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.LeakyReLU(inplace=True)
        )

        self.classifier_high = nn.Conv2d(self.inter_channel, n_class, kernel_size=1)
        self.classifier_low = nn.Conv2d(self.low_channel, n_class, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)

        # low channel branch
        x_low = self.classifier_low(features['low_channel'])

        # high channel branch
        x_high = self.CBR(features['high_channel'])
        s_w = self.squeeze_and_excitation(features['high_channel'])
        x_high = F.interpolate(x_high * s_w, size=[*x_low.shape[-2:]], mode='bilinear')
        x_high = self.classifier_high(x_high)

        return x_low + x_high

    @property
    def decoder_parameters(self):
        params = list()
        for m in [self.squeeze_and_excitation, self.CBR, self.classifier_low, self.classifier_high]:
            params.extend(list(m.parameters()))
        return params

    @property
    def num_classes(self):
        return self.classifier_low.out_channels

    @torch.no_grad()
    def feature_extraction(self, img: torch.Tensor, mode: str):
        """
        extract the feature for the classification head
        :param img:
        :param mode: mode of the classification head
        :return:
        """
        assert not self.training
        # normalization is done in dataset
        if mode == 'seg':
            features = self(img).softmax(1)
        elif mode == 'latent':
            features = self.encoder(img)['high_channel']
        else:
            raise ValueError('unknown mode')

        return features


class MultiSegLabelRASPP(LoadableModel):
    @store_config_args
    def __init__(self, n_class: int = 6, backbone: str = 'resnet18', spatial_dim: int = 124):
        super(MultiSegLabelRASPP, self).__init__()
        backbone_models = {'resnet18': models.resnet18}
        self.spatial_dim = spatial_dim  # 224  # due to pretraining on ImageNet

        self.low_channel = 128
        self.inter_channel = 128  # channel number of LR-ASPP head
        self.high_channel = 256

        self.encoder = models._utils.IntermediateLayerGetter(backbone_models[backbone](pretrained=True),
                                                             {'layer2': 'low_channel',
                                                              'layer3': 'high_channel'})
        self.z_space = nn.AdaptiveAvgPool2d(1)

        self.squeeze_and_excitation = nn.Sequential(
            nn.Conv2d(self.high_channel, out_channels=self.inter_channel, kernel_size=1, bias=False),
            # no batchnorm and bias like in the paper
            nn.Sigmoid()
        )

        self.airway_cnt_classifier = nn.Sequential(
            nn.Linear(self.high_channel, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, n_class, bias=True)
        )

        self.CBR = nn.Sequential(
            nn.Conv2d(self.high_channel, self.inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.LeakyReLU(inplace=True)
        )

        self.classifier_high = nn.Conv2d(self.inter_channel, n_class, kernel_size=1)
        self.classifier_low = nn.Conv2d(self.low_channel, n_class, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)

        # low channel branch
        x_low = self.classifier_low(features['low_channel'])

        # high channel branch
        x_high = self.CBR(features['high_channel'])
        z_vector = self.z_space(features['high_channel'])
        s_w = self.squeeze_and_excitation(z_vector)
        x_high = F.interpolate(x_high * s_w, size=[*x_low.shape[-2:]], mode='bilinear')
        x_high = self.classifier_high(x_high)

        # prediction heads
        seg_mask = x_high + x_low
        airway_cnt = self.airway_cnt_classifier(z_vector.flatten(-3, -1))

        return seg_mask, airway_cnt

    @property
    def decoder_parameters(self):
        params = list()
        for m in [self.squeeze_and_excitation, self.CBR, self.classifier_low, self.classifier_high,
                  self.airway_cnt_classifier]:
            params.extend(list(m.parameters()))
        return params

    @property
    def num_classes(self):
        return self.classifier_low.out_channels


if __name__ == '__main__':
    from torchinfo import summary

    m = LiteRASPP(spatial_dim=128)
    # m = MultiSegLabelRASPP(6)
    s = m.spatial_dim
    out = m(torch.randn(1, 3, s, s))
    summary(m, (1, 3, s, s))
    # m.save('/home/ron/Downloads/test.pt')
    # m1 = LiteRASPP.load('/home/ron/Downloads/test.pt', 'cpu')
    # print(m1.spatial_dim)
