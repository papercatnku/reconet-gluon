import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
from mxnet.gluon.model_zoo import vision
# import exception

# rgb preprocess
pre_mean=mx.nd.array((0.485, 0.456, 0.406)).reshape((1,3,1,1))
pre_std =mx.nd.array((0.229, 0.224, 0.225)).reshape((1,3,1,1))

cont_layer_name = [
    'relu3_2',
]

style_layer_name = [
    'relu1_2',
    'relu2_2',
    'relu3_3',
    'relu4_3',
]
def rgb_prerocess(x):
    return (x - pre_mean.as_in_context(x.context)) / pre_std.as_in_context(x.context)

class Vgg16(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Vgg16, self).__init__(prefix='')
        with self.name_scope():
            self.conv1_1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=1, padding=1)
            self.conv1_2 = nn.Conv2D(in_channels=64, channels=64, kernel_size=3, strides=1, padding=1)
            self.conv2_1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=1, padding=1)
            self.conv2_2 = nn.Conv2D(in_channels=128, channels=128, kernel_size=3, strides=1, padding=1)
            self.conv3_1 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=1, padding=1)
            self.conv3_2 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)
            self.conv3_3 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)
            self.conv4_1 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=1, padding=1)
            self.conv4_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
            self.conv4_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
    
    def hybrid_forward(self, F, X):
        h = F.Activation(self.conv1_1(X), act_type='relu')
        h = F.Activation(self.conv1_2(h), act_type='relu')
        relu1_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))
        h = F.Activation(self.conv2_1(h), act_type='relu')
        h = F.Activation(self.conv2_2(h), act_type='relu')
        relu2_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))
        h = F.Activation(self.conv3_1(h), act_type='relu')
        h = F.Activation(self.conv3_2(h), act_type='relu')
        relu3_2 = h
        h = F.Activation(self.conv3_3(h), act_type='relu')
        relu3_3 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))
        h = F.Activation(self.conv4_1(h), act_type='relu')
        h = F.Activation(self.conv4_2(h), act_type='relu')
        h = F.Activation(self.conv4_3(h), act_type='relu')
        relu4_3 = h
        
        return [relu3_3], [relu1_2, relu2_2, relu3_3, relu4_3]