import torch
from torchvision.models import resnet34, ResNet34_Weights
from torchsummary import summary
import torch.nn as nn
from convlstm import ConvLSTM



## El modelo que vamos a utilizar estará conformado con una ResNet-34 como encoder, una ConvLSTM en el neck
## y un decoder tipo U-Net. Utilizaremos inicialmente una ResNet-34 pre-entrenada en ImageNet, aunque probaremos si funciona
## mejor con dicho pre-entrenamiento y fine-tuneada con nuestros datos, o entrenada desde 0. ConvLSTM y el decoder sí serán
## entrenados desde 0 en ambos casos


p =  resnet34(weights = ResNet34_Weights.DEFAULT)


summary(p, (3,300,300))
# (64,4,1,300,300)


class encoderMet(nn.Module):
    def __init__(self,encoder_preentrenado = True):
        super().__init__()
        self.encoder_base = resnet34(weights = ResNet34_Weights.DEFAULT) if encoder_preentrenado else resnet34(weights = None)
        # Ahora vamos a actualizar la primera capa del modelo para que la entrada sea una imagen de 1 canal (las imagenes de radar tienen 1 canal solo)
        if encoder_preentrenado:
            pesos = self.encoder_base.conv1.weight
            peso_actu = pesos.mean(dim = 1, keepdim=True)
            self.encoder_base.conv1 = nn.Conv2d(1,64, kernel_size= (7,7), stride = (2,2), padding = (3,3), bias = False)
            with torch.no_grad():
                self.encoder_base.conv1.weight.copy_(peso_actu)
        else:
            self.encoder_base.conv1 = nn.Conv2d(1,64, kernel_size= (7,7), stride = (2,2), padding = (3,3), bias = False)
            nn.init.kaiming_normal_(self.encoder_base.conv1.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self,x):
        b,t,c,h,w = x.shape
        x = x.reshape(b*t,c,h,w) # Aplanamos la dimension temporal para poder pasarsela al encoder. Para pasarla al ConvLSTM y que sí trate la secuencia temporal lo reconstruimos a la salida del encoder
        x = self.encoder_base.conv1(x)
        x = self.encoder_base.bn1(x)
        x = self.encoder_base.relu(x)
        x = self.encoder_base.maxpool(x)
        skip1 = self.encoder_base.layer1(x)
        skip2 = self.encoder_base.layer2(skip1)
        skip3 = self.encoder_base.layer3(skip2)
        entrada_convlstm = self.encoder_base.layer4(skip3)
        # Los skips de arriba están preparados para inyectarse al decoder tipo U-Net que utilizaremos
        entrada_convlstm = entrada_convlstm.reshape(b,t,entrada_convlstm.shape[1],entrada_convlstm.shape[2], entrada_convlstm.shape[3])
        skip1 = skip1.reshape(b,t,skip1.shape[1],skip1.shape[2],skip1.shape[3]) # ahora mismo tienen forma 
        skip2 = skip2.reshape(b,t,skip2.shape[1],skip2.shape[2],skip2.shape[3])
        skip3 = skip3.reshape(b,t,skip3.shape[1],skip3.shape[2],skip3.shape[3])

        return entrada_convlstm, (skip1,skip2,skip3)


class neckMet(nn.Module):
    def __init__(self,h = 256,k = 3,n = 1):
        super().__init__()
        self.neck = ConvLSTM(input_dim= 512, hidden_dim = h, kernel_size=(k,k), num_layers= n , batch_first= True)
        self.adap = nn.Conv2d(h, 512, kernel_size= 1) # Conv 1x1 para adaptar la salida a 512 canales, con el objetivo de realizar concat con los skips
    
    def forward(self,x):
        _,last_states_list  = self.neck(x) 
        ht = last_states_list[-1][0] # Salida tendrá tamaño, con parámetros por defecto, de (B,256,10,10)
        ht = self.adap(ht) #(B,512,10,10)
        return ht



class decoderMet(nn.Module):
    def __init__(self):
        super().__init__()
        


        