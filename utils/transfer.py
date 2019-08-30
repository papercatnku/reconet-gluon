import mxnet as mx
from block import *
intround = lambda x:int(round(x))

class Reconet(nn.HybridBlock):
    def __init__(self, nf=1.0, ifin=True, ifinf=False):
        super(Reconet, self).__init__()
        self.encoder = nn.HybridSequential()
        self.decoder = nn.HybridSequential()

        with self.encoder.name_scope():
            self.encoder.add(
                ReconetConvBlock(
                channels=intround(nf*32),
                kernel=5,
                stride=1,
                bias=True,
                ifin=ifin,
                ifinf=ifinf),
                ReconetConvBlock(
                    channels=intround(nf*64),
                    kernel=3,
                    stride=2,
                    bias=True,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetConvBlock(
                    channels=intround(nf*128),
                    kernel=3,
                    stride=2,
                    bias=True,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetResidualBlock(
                    channels=intround(nf*128),
                    kernel=1,
                    stride=1,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetResidualBlock(
                    channels=intround(nf*128),
                    kernel=1,
                    stride=1,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetResidualBlock(
                    channels=intround(nf*128),
                    kernel=1,
                    stride=1,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetResidualBlock(
                    channels=intround(nf*128),
                    kernel=1,
                    stride=1,
                    ifin=ifin,
                    ifinf=ifinf)
                )
        with self.decoder.name_scope():
            self.decoder.add(
                ReconetUpSampleBlock(
                    channels=intround(nf*64),
                    kernel=3,
                    ifin=ifin,
                    ifinf=ifinf),
                ReconetUpSampleBlock(
                    channels=intround(nf*32),
                    kernel=3,
                    ifin=ifin,
                    ifinf=ifinf),
                ConvLayer(
                    channels=3,
                    kernel=5,
                    stride=1,
                    bias=True,
                    ifinf=ifinf),
                nn.HybridLambda(lambda F, x: F.sigmoid(x))
            )
        return
    
    def hybrid_forward(self, F, x):
        fm = self.encoder(x)
        out = self.decoder(fm)
        return out, fm

if __name__ == '__main__':

    ctx=mx.cpu()
    reconet = Reconet(nf=1, ifin=True, ifinf=False)
    reconet.hybridize()
    reconet.initialize(init=mx.initializer.Xavier(),ctx=ctx)
    dummy_in = mx.nd.random.normal(0,1,(4,3,360,640))

    dummy_out = reconet(dummy_in)
    print(dummy_out[0].shape)
    print(dummy_out[1].shape)

    reconet.export('reconet',0)

    reconet_sym = reconet(mx.sym.var('data'))[0]

    mx.viz.print_summary(
        reconet_sym,
        shape={
            'data':(1,3,360,640)
        }
    )