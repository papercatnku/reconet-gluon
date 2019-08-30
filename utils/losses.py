# coding: utf-8
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn

class TVLoss(gluon.loss.Loss):
    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(TVLoss, self).__init__(
            weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, x):
        x_ = F.slice_axis(x, axis=3, begin=1, end=None)
        _x = F.slice_axis(x, axis=3, begin=0, end=-1)
        y_ = F.slice_axis(x, axis=2, begin=1, end=None)
        _y = F.slice_axis(x, axis=2, begin=0, end=-1)
        x_grad = F.mean(F.abs(x_ - _x), axis=self._batch_axis, exclude=True)
        y_grad = F.mean(F.abs(y_ - _y), axis=self._batch_axis, exclude=True)
        return 0.5 * (x_grad + y_grad)

class CustomMSE(gluon.loss.Loss):
    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(CustomMSE, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, x, x_tar):
        se = F.square(x - x_tar) / 2.0
        mse = F.mean(se, axis=self._batch_axis, exclude=True)
        return mse

class CustomSSE(gluon.loss.Loss):
    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(CustomSSE, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, x, x_tar):
        se = F.square(x - x_tar) / 2.0
        mse = F.sum(se, axis=self._batch_axis, exclude=True)
        return mse

RGB_LUMIN_FACTOR = mx.nd.reshape(
    mx.nd.array([0.2126, 0.7152, 0.0722],ctx=mx.cpu()),
    shape=(1,3,1,1))

def Image2Luminace(x):
    luminance = x * RGB_LUMIN_FACTOR.as_in_context(x.context)
    luminance = mx.nd.sum(luminance, axis=1,keepdims=True)
    return luminance

def resize_flow(flow, width, height):
    ori_height = flow.shape[1]
    ori_width = flow.shape[2]
    rsz_flw = mx.nd.contrib.BilinearResize2D(
        flow,
        height=height,
        width=width)
    rsz_flw[:,0,:,:] *= height / ori_height
    rsz_flw[:,1,:,:] *= width / ori_width
    return rsz_flw


def warp(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    gg_data = mx.nd.GridGenerator(data=flow, transform_type='warp')
    warp_out = mx.nd.BilinearSampler(data=x, grid=gg_data)
    return warp_out

class LuminanceLoss(gluon.loss.Loss):
    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(LuminanceLoss, self).__init__(weight, batch_axis, **kwargs)
        return
    def hybrid_forward(self, F, x, xlast, y, ylast, flow, mask):
        _,channels,h,w = x.shape
        xluminchange = Image2Luminace(x - warp(xlast, flow))
        warperr = y - warp(ylast, flow)
        channel_coherent_err =mask * mx.nd.square(warperr - xluminchange)
        loss = mx.nd.sum(channel_coherent_err, axis=self._batch_axis, exclude=True) / float(h*w)
        return loss

class FeatureLoss(gluon.loss.Loss):
    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        super(FeatureLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, x, xlast, flow, mask):
        _,channels,h,w = x.shape
        rsz_flow = resize_flow(flow, w, h)
        rsz_mask = F.contrib.BilinearResize2D(
            mask,
            width=w,
            height=h)
        warpres = warp(xlast, rsz_flow)
        loss = mx.nd.sum(mx.nd.square(x-warpres) * rsz_mask, axis=self._batch_axis, exclude=True) / float(h*w)
        return loss