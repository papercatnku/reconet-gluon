import base64
import sys
import time
from time import localtime
import os
import logging
import cv2
from IPython.display import HTML, Image, clear_output, display
import mxnet as mx

def arrayShow(imageArray):
    ret, png = cv2.imencode('.png', imageArray)
    encoded = base64.b64encode(png)
    return Image(data=encoded.decode('ascii'))

def output_process_status(total_percent, num_bar = 40):
    process_bar = '['+ '#'*int(round(total_percent * num_bar)) + ' ' * (num_bar-int(round(total_percent * num_bar))) + ']'
    lineout = '\rprocessing: %s  %.0f%%'%(process_bar, total_percent * 100)
    sys.stdout.write(lineout)
    return

def gettimetag():
    curtime = localtime()
    time_prefx = "%d%.2d%.2d%.2d%.2d%.2d" % (
        curtime.tm_year - 2000,
        curtime.tm_mon,
        curtime.tm_mday,
        curtime.tm_hour,
        curtime.tm_min,
        curtime.tm_sec)
    return time_prefx

def setuplogger(modelpath):
    model_dir, model_fn = os.path.split(modelpath)
    if model_dir == '':
        model_dir = './'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    log_fn = os.path.join(model_dir, model_fn +'.log')
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    # file log
    fh = logging.FileHandler(log_fn)
    fh.setFormatter(formatter) 
    # consolo log
    ch = logging.StreamHandler(sys.stdout)
    ch.formatter = formatter 
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def makeStyleImageTensor(srcimg, wh=(768, 768), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    rsz_img = mx.image.imresize(
        srcimg, w=wh[0], h=wh[1], interp=cv2.INTER_LANCZOS4)
    tensor_img = mx.nd.image.to_tensor(rsz_img)
    normed_img = mx.nd.image.normalize(tensor_img, mean=mean, std=std)
    normed_img_4d = mx.nd.expand_dims(normed_img, axis=0)
    return normed_img_4d