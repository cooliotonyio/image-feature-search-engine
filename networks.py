import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.penult_layer = self.net._modules.get('avgpool')
        self.net.eval()
    
    def forward(self, x):
        output = self.get_embedding(self, x)
        return output
    
    def get_embedding(self, x):
        embedding = torch.cuda.FloatTensor(x.shape[0], 512, 1, 1).fill_(0)
        def copy(m, i ,o):
            embedding.copy_(o.data)
        hook = self.penult_layer.register_forward_hook(copy)
        self.net(x)
        hook.remove()
        return embedding.view(embedding.size()[0], -1)