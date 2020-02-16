import os
import sys

from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from networks.convnet import ConvNet


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = ConvNet(z_dim=args.dim)

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
            logits = -dist_matrix(data_query, proto, c=self.e2p.c) / self.args.temperature

        else:
            if self.training:
                proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            else:
                proto = proto.reshape(self.args.shot, self.args.validation_way, -1).mean(dim=0)

            logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
        return logits
