import torch
import torch.nn as nn
import torch.nn.parallel
from torchvision import models
from options.train_options import TrainOptions
import os

opt = TrainOptions().parse()


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False), )
        else:
            self.block = nn.Sequential(nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False), norm_layer(in_features), nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False), norm_layer(in_features))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class MISS(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, input_dim=8, mask_dim=5, activ='softmax'):
        super(MISS, self).__init__()
        self.mask_dim = mask_dim
        self._name = 'resunet_generator'
        self.activ = activ
        self.encoder_channels = [64, 128, 256, 512]
        self.decoder_channels = [512, 256, 128, 64]
        self.encoders = []
        input_nc = input_dim
        self.encoders.append(nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(inplace=True), ResidualBlock(64), ResidualBlock(64)))
        input_nc = 64
        for inner_nc in self.encoder_channels[1:]:
            self.encoders.append(nn.Sequential(nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(inner_nc), nn.ReLU(inplace=True),
                ResidualBlock(inner_nc), ResidualBlock(inner_nc)))
            input_nc = inner_nc
        self.encoders.append(nn.Sequential(nn.Conv2d(input_nc, 512, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(inplace=True), ResidualBlock(512), ResidualBlock(512)))
        self.encoders = nn.Sequential(*self.encoders)

        # Up-Sampling
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoders_mask = []
        self.decoders_mask.append(
            nn.Sequential(upsample, nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), ResidualBlock(512),
                          ResidualBlock(512)))

        for inner_nc in self.decoder_channels[1:]:

            self.decoders_mask.append(
                nn.Sequential(upsample, nn.Conv2d(input_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(inner_nc), nn.ReLU(inplace=True),
                    ResidualBlock(inner_nc), ResidualBlock(inner_nc)))
            input_nc = inner_nc
        self.decoders_mask.append(nn.Sequential(upsample, nn.Conv2d(input_nc * 2, mask_dim, kernel_size=3, stride=1, padding=1, bias=False)))
        self.decoders_mask = nn.Sequential(*self.decoders_mask)

        self.decoders_content = []
        self.decoders_content.append(
            nn.Sequential(upsample, nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), ResidualBlock(512),
                          ResidualBlock(512)))
        input_nc = 512
        for inner_nc in self.decoder_channels[1:]:

            self.decoders_content.append(
                nn.Sequential(upsample, nn.Conv2d(input_nc * 2, inner_nc, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(inner_nc), nn.ReLU(inplace=True),
                    ResidualBlock(inner_nc), ResidualBlock(inner_nc)))
            input_nc = inner_nc
        self.decoders_content.append(nn.Sequential(upsample, nn.Conv2d(input_nc * 2, (mask_dim - 1) * 3, kernel_size=3, stride=1, padding=1, bias=False)))
        self.decoders_content = nn.Sequential(*self.decoders_content)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        encoder_outs = self.encode(x)
        if self.activ == 'softmax':
            d_out_mask = self.softmax(self.decode(encoder_outs, self.decoders_mask))
        elif self.activ == 'sigmoid':
            d_out_mask = self.sigmoid(self.decode(encoder_outs, self.decoders_mask))
        else:
            raise NotImplementedError

        d_out_content = torch.tanh(self.decode(encoder_outs, self.decoders_content))
        attention_mask_list = []
        content_list = []
        for i in range(self.mask_dim - 1):
            attention_mask_list.append(d_out_mask[:, i:i+1, :, :])
            content_list.append(d_out_content[:, i*3:(i+1)*3, :, :])
        attention_mask_list.append(d_out_mask[:, self.mask_dim-1:self.mask_dim, :, :])
        outputs = []
        for i in range(self.mask_dim - 1):
            outputs.append(content_list[i] * attention_mask_list[i])
        outputs.append(x[:,0:3,:,:] * attention_mask_list[-1])
        res_image = outputs[0]
        for i in range(1, self.mask_dim):
            res_image += outputs[i]
        return res_image, attention_mask_list, content_list, outputs
    def encode(self, x):
        encoder_outs = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            encoder_outs.append(x)  # print(i, x.shape)
        return encoder_outs

    def decode(self, encoder_outs, decoders):
        x = encoder_outs[-1]
        x = decoders[0](x)
        for i in range(1, len(self.encoders)):
            x = torch.cat([encoder_outs[4 - i], x], 1)
            x = decoders[i](x)  # x * 2
        return x

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x.contiguous()), self.vgg(y.contiguous())
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def warp(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        loss += self.weights[4] * self.criterion(x_vgg[4], y_vgg[4].detach())
        return loss


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!: {}'.format(checkpoint_path))
        return

    checkpoint = torch.load(checkpoint_path)

    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)
