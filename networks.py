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


class EmbeddingNet(nn.Module):
    def __init__(self, resnet = None):
        super(EmbeddingNet, self).__init__()
        if resnet is None:
            resnet = models.resnet18(pretrained=True)
            
        self.resnet = resnet
        self.resnet_layer = self.resnet._modules.get('avgpool')
        self.resnet.eval()

        self.fc = nn.Sequential(nn.Linear(64 * 8, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 16)
                                )
    def forward(self, x):
        output = self.get_resnet_embedding(x)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
    def get_resnet_embedding(self, x):
        embedding = torch.cuda.FloatTensor(x.shape[0],512,1,1).fill_(0)
        def copy(m, i, o):
            embedding.copy_(o.data)
        hook = self.resnet_layer.register_forward_hook(copy)
        self.resnet(x)
        hook.remove()
        return embedding.view(embedding.size()[0], -1)

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(16, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1,output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)