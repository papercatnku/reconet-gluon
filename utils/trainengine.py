import os
from timeit import default_timer as timer

import cv2
import mxnet as mx
import mxnet.autograd as ag
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import numpy as np
from mxnet.gluon.model_zoo import vision
from math import sqrt

from block import VGG_MEAN, VGG_STD, VGGStandardize, gram_matrix
from descriptor import Vgg16, cont_layer_name, rgb_prerocess, style_layer_name
from losses import CustomMSE, CustomSSE, FeatureLoss, LuminanceLoss, TVLoss
from miscellaneous import *

default_training_params = {
    'stylepixels': 640*360,
    'style_w': 10,
    'content_w': 1.0,
    'tv_w': 1.0,
    # 'out_temporal_w':1e7,
    'out_temporal_w':10.0,
    # 'fm_temporal_w':2e3,
    'fm_temporal_w':1.0,
    'batch_interval': 100,
    'internal_check_dir': './check-result-0828',
    'epoch': 10,
    'epoch_interval': 1
}

def whUniformScale(src_wh, pixel_limits):
    w, h = src_wh
    r = sqrt(float(pixel_limits) / float(np.prod(src_wh)))

    new_w = (int((w * r) + 4) // 8) * 8
    new_h = (int((h * r) + 4) // 8) * 8
    return new_w, new_h

class ReconetTrain:
    def _init_metric(self,):
        self.style_metric_train_ls = [mx.metric.Loss(x) for x in style_layer_name]
        for _met in self.style_metric_train_ls:
            _met.reset()
        self.cont_metric_train_ls = [mx.metric.Loss(x) for x in cont_layer_name]
        for _met in self.cont_metric_train_ls:
            _met.reset()
        self.tv_metric =  mx.metric.Loss('tv')
        self.tv_metric.reset()
        self.frame_coherent_metric= mx.metric.Loss('frame-coherent')
        self.frame_coherent_metric.reset()
        self.fm_coherent_metric = mx.metric.Loss('fm-coherent')
        self.fm_coherent_metric.reset()

    def __init__(self, transfer, descriptor, optimizer, wh, ctx, logger):
        self._init_metric()
        self.transfer = transfer
        self.descriptor = descriptor
        self.optimizer = optimizer
        self.gram_targets = []
        self.wh = wh
        # losses
        self.content_loss = CustomMSE()
        self.style_loss = CustomSSE()
        self.tv_loss = TVLoss()
        self.frame_coherent_loss = LuminanceLoss()
        self.fm_coherent_loss = FeatureLoss()
        self.ctx = ctx
        self.logger = logger

        self.train_tic = timer()
        return

    def extract_grams(self, styleimg, stylelimits=230400, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        wh = (styleimg.shape[1], styleimg.shape[0])
        wh4gramtar = whUniformScale(wh, stylelimits)
        style_tensor = makeStyleImageTensor(
            styleimg, wh4gramtar, mean, std)
        style_tensor = style_tensor
        _, style_fms = self.descriptor(style_tensor.as_in_context(self.ctx))
        grams = list(map(gram_matrix, style_fms))
        return grams

    def tsr2bgrimg(self, tsf_tensor):
        return

    def train(
        self,
        style_img_fn,
        dataloader_train,
        dataloader_eval,
        training_params,
        output_prefix):
        # gram targets
        styimg = mx.image.imread(style_img_fn)
        self.logger.info(
            'Extracting the gram matrix of image: %s.', style_img_fn)
        self.gram_targets = self.extract_grams(styimg, training_params['stylepixels'])
        self.logger.info(
            'The gram matrix of image: %s successfully extracted.', style_img_fn)
        if not os.path.exists(training_params['internal_check_dir']):
            os.makedirs(training_params['internal_check_dir'])
        
        for e in range(training_params['epoch']):
            #
            for i, srcbatch in enumerate(dataloader_train):
                # train
                batch_size = srcbatch[0].shape[0]
                x_cur, x_last, occ_mask, flow = srcbatch

                x_cur = x_cur.as_in_context(self.ctx)
                x_last = x_last.as_in_context(self.ctx)
                occ_mask = occ_mask.as_in_context(self.ctx)
                flow=flow.as_in_context(self.ctx)
                # 
                x_cur_cont= self.descriptor(VGGStandardize(x_cur))[0]
                x_last_cont = self.descriptor(VGGStandardize(x_last))[0]
                with ag.record():
                    y_cur, y_fm_cur = self.transfer(x_cur)
                    y_last, y_fm_last = self.transfer(x_last)
                    y_cur_cont, y_cur_sty = self.descriptor(VGGStandardize(y_cur))
                    y_last_cont, y_last_sty = self.descriptor(VGGStandardize(y_last))

                    cont_loss_ls = []
                    # content loss
                    for _y_cur_cont_fm, _y_last_cont_fm, \
                        _x_cur_cont_fm, _x_last_cont_fm in zip(
                            y_cur_cont,
                            y_last_cont,
                            x_cur_cont,
                            x_last_cont):
                        _fm_loss = (
                            training_params['content_w'] * self.content_loss(
                                _y_cur_cont_fm, _x_cur_cont_fm) + 
                            training_params['content_w'] * self.content_loss(
                                _y_last_cont_fm, _x_last_cont_fm)
                        )
                        cont_loss_ls.append(_fm_loss)
                    # style loss
                    sty_loss_ls = []
                    for _y_cur_sty_fm, _y_last_sty_fm, _gram_tar in zip(
                        y_cur_sty, y_last_sty, self.gram_targets):
                        _y_cur_gram = gram_matrix(_y_cur_sty_fm)
                        _y_last_gram = gram_matrix(_y_last_sty_fm)
                        _gram_loss = (
                            training_params['style_w'] * self.style_loss(
                                _y_cur_gram, _gram_tar) +
                            training_params['style_w'] * self.style_loss(
                                _y_last_gram, _gram_tar)
                        )
                        sty_loss_ls.append(_gram_loss)
                    # tv_loss
                    tv_loss = (
                        training_params['tv_w'] * self.tv_loss(y_cur) + 
                        training_params['tv_w'] * self.tv_loss(y_last))
                    # image_lumin_loss
                    # image_lumin_loss = training_params['out_temporal_w'] * self.frame_coherent_loss(
                    #     x_cur, x_last, y_cur, y_last, flow, occ_mask)
                    # fm_loss
                    fm_loss =(
                        training_params['fm_temporal_w'] * self.fm_coherent_loss(
                            y_fm_cur, y_fm_last, flow, occ_mask)
                    )
                    # total_loss = (
                    #     sum(cont_loss_ls) + sum(sty_loss_ls) + tv_loss + 
                    #     image_lumin_loss + fm_loss)
                    total_loss = (
                        sum(cont_loss_ls) + sum(sty_loss_ls) + tv_loss + fm_loss
                        )
                total_loss.backward()
                self.optimizer.step(batch_size)
                mx.nd.waitall()
                #update metric
                for _ct_loss, _metric in zip(
                    cont_loss_ls, self.cont_metric_train_ls):
                    _metric.update(None, _ct_loss)
                for _sty_loss, _metric in zip(
                    sty_loss_ls, self.style_metric_train_ls
                ):
                    _metric.update(None, _sty_loss)
                # TV metric
                self.tv_metric.update(None, tv_loss)
                # frame coherent 
                # self.frame_coherent_metric.update(None, image_lumin_loss)
                self.fm_coherent_metric.update(None, fm_loss)

                # checkpoint
                if (i % default_training_params['batch_interval']==0):
                    # log
                    logstr = 'Training: Epoch[%d], Batch[%d],\n'
                    logparam = [e, i, ]
                    for _metric in self.cont_metric_train_ls:
                        for name, val in _metric.get_name_value():
                            logstr += '%s = %.3f,\n'
                            logparam.append(name)
                            logparam.append(val)
                    for _metric in self.style_metric_train_ls:
                        for name, val in _metric.get_name_value():
                            logstr += '%s = %.3f,\n'
                            logparam.append(name)
                            logparam.append(val)
                    logstr += '%s = %.3f,\n'
                    logparam.append(self.tv_metric.get_name_value()[0][0])
                    logparam.append(self.tv_metric.get_name_value()[0][1])
                    logstr += '%s = %.3f,\n'
                    logparam.append(self.frame_coherent_metric.get_name_value()[0][0])
                    logparam.append(self.frame_coherent_metric.get_name_value()[0][1])
                    logstr += '%s = %.3f,\n'
                    logparam.append(self.fm_coherent_metric.get_name_value()[0][0])
                    logparam.append(self.fm_coherent_metric.get_name_value()[0][1])

                    tc = timer() - self.train_tic
                    logstr += 'Time cost=%.3fms/batch, %.3f batch/s.'
                    logparam.append(tc * 1000 / training_params['batch_interval'])
                    logparam.append(training_params['batch_interval'] / tc)
                    self.logger.info(logstr, *logparam)

                    for _metric in self.cont_metric_train_ls:
                        _metric.reset()
                    for _metric in self.cont_metric_train_ls:
                        _metric.reset()
                    self.tv_metric.reset()
                    self.frame_coherent_metric.reset()
                    self.fm_coherent_metric.reset()
                    self.train_tic = timer()
                output_process_status(
                    float(i%training_params['batch_interval']) / float(training_params['batch_interval']),)
            if (e%training_params['epoch_interval'] == 0):
                self.transfer.export(path=output_prefix, epoch=e)
        return

    
if __name__ == '__main__':
    from dataloader import MPIDataset, FlyingthingsDataset, JointDataset, bfn4reconet
    from descriptor import Vgg16
    from transfer import Reconet
    

    logger = setuplogger('./log-test')
    ctx = mx.gpu(7)
    batch_size = 2
    wh = (640,360)

    tsf = Reconet(
        nf=1.0,
        ifin=True,
        ifinf=False)
    tsf.hybridize()
    tsf.initialize(
            init=mx.init.Xavier(magnitude=1.0),
            ctx=ctx,
            verbose=False)
    
    des = Vgg16(ctx=ctx)
    des.collect_params().load(
        './models/vgg16-64946753.params',
        allow_missing=False,
        ignore_extra=True,
        ctx=ctx)
    for param in des.collect_params().values():
        param.grad_req = 'null'

    trainer = gluon.Trainer(
        tsf.collect_params(),
        optimizer='adam',
        optimizer_params={
            'learning_rate': 0.0001,
            'wd': 0.0001,
            })
    
    style_img_fn = '/home/zcy6735/github/reconet-gluon/style/candy.jpg'

    engine = ReconetTrain(
        transfer=tsf,
        descriptor=des,
        optimizer=trainer,
        wh=wh,
        ctx=ctx,
        logger=logger)
    
    # ds_train = MPIDataset(
    #     '/home/zcy6735/dataset/MPI',
    #     wh=(640,360))
    ds_train = JointDataset(
        mpi_dirpath='/home/zcy6735/dataset/MPI',
        flyingthings_dirpath='/mnt/md0/Datasets/FlyingThings3D_subset/FlyingThings3D_subset',
        wh=wh
    )
    dl_train = gluon.data.dataloader.DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        last_batch='discard',
        batchify_fn=bfn4reconet)
    
    engine.train(
        style_img_fn=style_img_fn,
        dataloader_train=dl_train,
        dataloader_eval=None,
        training_params=default_training_params,
        output_prefix='./models/temptest/candy')
