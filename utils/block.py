import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn

VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def VGGStandardize(src):
    mean_nd = mx.nd.reshape(mx.nd.array(VGG_MEAN, ctx=src.context),shape=(1,3,1,1))
    norm_nd = mx.nd.reshape(mx.nd.array(VGG_STD, ctx=src.context),shape=(1,3,1,1))
    return (src - mean_nd) / norm_nd

def gram_matrix(x):
    (b, ch, h, w) = x.shape
    features = x.reshape((b, ch, w * h))
    gram = mx.nd.batch_dot(features, features, transpose_b=True) / (ch * h * w)
    return gram

# reconet blocks
class ConvLayer(nn.HybridBlock):
    def __init__(
        self,
        channels,
        kernel,
        stride,
        bias=True,
        ifinf=False,
        *args, **kwargs):
        super(ConvLayer, self).__init__(*args, **kwargs)
        with self.name_scope():
            if ifinf:
                self.pad =None
                self.conv = nn.Conv2D(
                    channels,
                    kernel,
                    strides=stride,
                    padding=kernel//2,
                    use_bias=bias)
            else:
                self.pad = nn.HybridLambda(
                    lambda F, x: F.pad(
                        x, 
                        mode='reflect', 
                        pad_width=(0, 0, 0, 0, 
                        kernel // 2, kernel // 2, kernel // 2, kernel // 2)))
                self.conv = nn.Conv2D(
                    channels,
                    kernel,
                    strides=stride,
                    padding=0,
                    use_bias=bias)
        return
    
    def hybrid_forward(self, F, x):
        if self.pad:
            x = self.pad(x)
        out = self.conv(x)
        return out

def ReconetConvBlock(channels, kernel, stride, bias=True, ifin=True, ifinf=False):
    out = nn.HybridSequential()
    out.add(
        ConvLayer(
            channels,
            kernel,
            stride,
            bias,
            ifinf=ifinf),
        nn.InstanceNorm(scale=True) if ifin else nn.BatchNorm(scale=True),
        nn.Activation('relu')
    )
    return out

class ReconetResidualBlock(nn.HybridBlock):
    def __init__(
        self,
        channels,
        kernel=1,
        stride=1,
        ifin=True,
        ifinf=False,
        *args, **kwargs):
        super(ReconetResidualBlock, self).__init__(*args, **kwargs)
        with self.name_scope():
            self.conv1 = ConvLayer(channels, kernel, stride,bias=False,ifinf=ifinf)
            self.norm1 = nn.InstanceNorm() if ifin else nn.BatchNorm()
            self.conv2 = ConvLayer(channels, kernel, stride,bias=False,ifinf=ifinf)
            self.norm2 = nn.InstanceNorm() if ifin else nn.BatchNorm()
            self.act = nn.Activation('relu')
        
    def hybrid_forward(self, F, x):
        residual = self.act(self.norm1(self.conv1(x)))
        residual = self.norm2(self.conv2(residual))
        return x + residual
        
class ReconetUpSampleBlock(nn.HybridBlock):
    def __init__(
        self, 
        channels,
        kernel,
        ifin=True,
        ifinf=False, 
        *args, **kwargs):
        super(ReconetUpSampleBlock, self).__init__(*args, **kwargs)
        with self.name_scope():
            self.conv =ConvLayer(
                channels,
                kernel,
                1,
                bias=True,
                ifinf=ifinf)
            self.norm = nn.InstanceNorm() if ifin else nn.BatchNorm()
            self.act = nn.Activation('relu')
        return
    
    def hybrid_forward(self, F, x):
        out = F.UpSampling(x, scale=2, sample_type='nearest')
        out = self.act(self.norm(self.conv(out)))
        return out

