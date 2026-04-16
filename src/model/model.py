import torch
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F
import torch.nn as nn
from .convlstm import ConvLSTM


class encoderMet(nn.Module):
    def __init__(self, encoder_preentrenado=True):
        super().__init__()
        self.encoder_base = resnet34(weights=ResNet34_Weights.DEFAULT) if encoder_preentrenado else resnet34(weights=None)
        if encoder_preentrenado:
            pesos = self.encoder_base.conv1.weight
            peso_actu = pesos.mean(dim=1, keepdim=True)
            self.encoder_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            with torch.no_grad():
                self.encoder_base.conv1.weight.copy_(peso_actu)
        else:
            self.encoder_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            nn.init.kaiming_normal_(self.encoder_base.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b*t, c, h, w)
        x = self.encoder_base.conv1(x)
        x = self.encoder_base.bn1(x)
        skip1 = self.encoder_base.relu(x)
        x = self.encoder_base.maxpool(skip1)
        skip2 = self.encoder_base.layer1(x)
        skip3 = self.encoder_base.layer2(skip2)
        skip4 = self.encoder_base.layer3(skip3)
        entrada_convlstm = self.encoder_base.layer4(skip4)
        entrada_convlstm = entrada_convlstm.reshape(b, t, entrada_convlstm.shape[1], entrada_convlstm.shape[2], entrada_convlstm.shape[3])
        skip1 = skip1.reshape(b, t, skip1.shape[1], skip1.shape[2], skip1.shape[3])
        skip2 = skip2.reshape(b, t, skip2.shape[1], skip2.shape[2], skip2.shape[3])
        skip3 = skip3.reshape(b, t, skip3.shape[1], skip3.shape[2], skip3.shape[3])
        skip4 = skip4.reshape(b, t, skip4.shape[1], skip4.shape[2], skip4.shape[3])
        return entrada_convlstm, (skip1, skip2, skip3, skip4)


class neckMet(nn.Module):
    def __init__(self, h=256, k=3, n=1, dropout_p=0.1):
        super().__init__()
        self.neck    = ConvLSTM(input_dim=512, hidden_dim=h, kernel_size=(k,k), num_layers=n, batch_first=True)
        self.adap    = nn.Conv2d(h, 512, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        _, last_states_list = self.neck(x)
        ht = last_states_list[-1][0]
        ht = self.adap(ht)
        ht = self.dropout(ht)
        return ht


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              stride=stride, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class decoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ConvModule(in_channels=out_channels + skip_channels, out_channels=out_channels),
            nn.Dropout2d(p=dropout_p),
            ConvModule(in_channels=out_channels, out_channels=out_channels),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SkipConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, k=3, num_layers=1, dropout_p=0.1):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                                  kernel_size=(k,k), num_layers=num_layers, batch_first=True)
        self.conv1    = (nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1, bias=False)
                         if hidden_dim != input_dim else nn.Identity())
        self.dropout  = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        _, last_state_list = self.convlstm(x)
        x = last_state_list[-1][0]
        x = self.conv1(x)
        x = self.dropout(x)
        return x


class decoderMet(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dec1 = decoderBlock(in_channels=512,  skip_channels=256, out_channels=256, dropout_p=dropout_p)
        self.dec2 = decoderBlock(in_channels=256,  skip_channels=128, out_channels=128, dropout_p=dropout_p)
        self.dec3 = decoderBlock(in_channels=128,  skip_channels=64,  out_channels=64,  dropout_p=dropout_p)
        self.dec4 = decoderBlock(in_channels=64,   skip_channels=64,  out_channels=64,  dropout_p=dropout_p)

        self.skipLSTM1 = SkipConvLSTM(input_dim=64,  hidden_dim=32,  dropout_p=dropout_p)
        self.skipLSTM2 = SkipConvLSTM(input_dim=64,  hidden_dim=32,  dropout_p=dropout_p)
        self.skipLSTM3 = SkipConvLSTM(input_dim=128, hidden_dim=64,  dropout_p=dropout_p)
        self.skipLSTM4 = SkipConvLSTM(input_dim=256, hidden_dim=128, dropout_p=dropout_p)

        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                  ConvModule(64, 64))
        self.head = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x, skips, original_shape):
        skip1, skip2, skip3, skip4 = skips
        skip1 = self.skipLSTM1(skip1)
        skip2 = self.skipLSTM2(skip2)
        skip3 = self.skipLSTM3(skip3)
        skip4 = self.skipLSTM4(skip4)
        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)
        x = self.up1(x)
        x = F.interpolate(x, size=original_shape, mode="bilinear", align_corners=False)
        return self.head(x)


class modelMet(nn.Module):
    def __init__(self, encoder_preentrenado=True, h=256, k=3, n=1, dropout_p=0.1):
        super().__init__()
        self.encoder = encoderMet(encoder_preentrenado)
        self.neck    = neckMet(h=h, k=k, n=n, dropout_p=dropout_p)
        self.decoder = decoderMet(dropout_p=dropout_p)

    def forward(self, x):
        _, _, _, H, W = x.shape
        entrada_convlstm, skips = self.encoder(x)
        ht  = self.neck(entrada_convlstm)
        out = self.decoder(ht, skips, (H, W))
        return out
