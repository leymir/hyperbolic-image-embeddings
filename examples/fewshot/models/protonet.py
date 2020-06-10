import os
import sys

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from networks.convnet import ConvNet
from networks.ResNet import resnet18, resnet34
from networks.ResNet12 import Res12
from networks.DenseNet import densenet121
from networks.WideResNet import wideres


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.model

        if model_name == "convnet":
            self.encoder = ConvNet(z_dim=args.dim)
        elif model_name == "resnet18":
            self.encoder = resnet18(remove_linear=True)
        elif model_name == "resnet34":
            self.encoder = resnet34(remove_linear=True)
        elif model_name == "densenet121":
            self.encoder = densenet121(remove_linear=True)
        elif model_name == "wideres":
            self.encoder = wideres(remove_linear=True)
        elif model_name == "resnet12":
            self.encoder = Res12()
        else:
            raise ValueError("Model not found")

        if args.hyperbolic:
            self.e2p = ToPoincare(c=args.c, train_c=args.train_c, train_x=args.train_x)

    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        if self.args.hyperbolic:
            proto = self.e2p(proto)

            if self.training:
                proto = proto.reshape(self.args.shot, self.args.way, -1)
            else:
                proto = proto.reshape(self.args.shot, self.args.validation_way, -1)

            proto = poincare_mean(proto, dim=0, c=self.e2p.c)
            data_query = self.e2p(self.encoder(data_query))
            logits = (
                -dist_matrix(data_query, proto, c=self.e2p.c) / self.args.temperature
            )

        else:
            if self.training:
                proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            else:
                proto = proto.reshape(
                    self.args.shot, self.args.validation_way, -1
                ).mean(dim=0)

            logits = (
                euclidean_metric(self.encoder(data_query), proto)
                / self.args.temperature
            )
        return logits
