import argparse
import glob
import json
import math
import os
import random
import subprocess
import time
import zipfile
import multiprocessing as mp

try:
    subprocess.run(["python", "-m", "pip", "install", "gluoncv"])
    import gluoncv as gcv
    from gluoncv.utils import viz
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
    from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
except:
    print("Cannot install dependencies")
        

import mxnet as mx
from mxnet import gluon, nd, autograd
import numpy as np



def get_dataloader(net, train_dataset,validation_dataset, image_h, image_w, batch_size, num_workers):
    """
    Get dataloader.
    """

    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated

    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLO3DefaultTrainTransform(image_w, image_h, net)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover',num_workers=num_workers,prefetch=num_workers+num_workers//2)
    
    val_transform = YOLO3DefaultValTransform(image_w, image_h)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        validation_dataset.transform(val_transform), 
        batch_size, False, batchify_fn=batchify_fn, last_batch='keep',num_workers=num_workers,prefetch=num_workers+num_workers//2)

    return train_loader, val_loader

def parse_args():
    """
    Get the arguments
    """
    
    parser = argparse.ArgumentParser()


    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--model', type=str, default="yolo3_mobilenet0.25_custom")
    parser.add_argument('--upscale-size',type=int,default=208)
    

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()

    
def validate(net, val_data, ctx, classes,metric):
    """
    Compute the mAP for the network on the validation data
    """
    mx.nd.waitall()
    net.hybridize()
    metric.reset()
    for ib, batch in enumerate(val_data):

        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes, det_ids, det_scores = [],[],[]
        gt_bboxes,gt_ids = [], []

        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
    return metric.get()

if __name__ == '__main__':
    
    args = parse_args()

    ########################
    # Parameters           #
    ########################
    image_h = args.upscale_size
    image_w = args.upscale_size
    ctx = [mx.gpu(i) for i in range(mx.context.num_gpus())] if mx.context.num_gpus() > 0 else [mx.cpu()]
    print(ctx)
    batch_size = args.batch_size*len(ctx)
    num_epochs = args.epochs
    num_workers = mp.cpu_count()-1 if mp.cpu_count()>1 else 0
    num_workers = num_workers//2
    print('num worksers is {}'.format(num_workers))
    learning_rate = args.lr
    learning_rate_factor = args.lr_factor
    wd = args.wd
    
    classes = ['Passenger Vehicle',
                'Small Car',
                'Bus',
                'Pickup Truck',
                'Utility Truck',
                'Truck',
                'Cargo Truck',
                'Truck Tractor w/ Box Trailer',
                'Truck Tractor',
                'Trailer',
                'Truck Tractor w/ Flatbed Trailer',
                'Truck Tractor w/ Liquid Tank']
    
    ########################
    # Data                 #
    ########################

    train_dataset = gcv.data.RecordFileDetection(os.path.join(args.train,'train.rec'),coord_normalized=False)
    validation_dataset = gcv.data.RecordFileDetection(os.path.join(args.train,'val.rec'),coord_normalized=False)
    
    print("There is {} training images, {} validation images".format(len(train_dataset), len(validation_dataset)))
    
    ########################
    # Network              #
    ########################
    net = gcv.model_zoo.get_model(args.model, pretrained=False,classes=classes)
    net.initialize(force_reinit=True,ctx=ctx)
    train_data, val_data = get_dataloader(net, train_dataset,validation_dataset, image_h, image_w, batch_size, num_workers)
     
    net.hybridize()
    
    ########################
    # Optimizer            #
    ########################
    steps_epochs = [num_epochs / 3, (2*num_epochs)/3]
    iterations_per_epoch = math.ceil(len(train_dataset) / batch_size)
    steps_iterations = [int(s*iterations_per_epoch) for s in steps_epochs]
    print("Learning rate drops after iterations: {}".format(steps_iterations))
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=learning_rate_factor)

    trainer = gluon.Trainer(
        net.collect_params(), 'Nadam',
        {'learning_rate': learning_rate, 'wd': wd, 'lr_scheduler':schedule})
    
    metric = VOC07MApMetric(iou_thresh=0.3, class_names=classes)
    net.set_nms(nms_thresh=0.45, nms_topk=100, post_nms=100)
    
    best_mAP = 0
    for epoch in range(num_epochs):
        mx.nd.waitall()
        net.hybridize(static_alloc=True,static_shape=True)
        tic = time.time()
        btic = time.time()
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            btic = time.time()
    
        
        mAP = validate(net, val_data, ctx, classes,metric)[1][-1]
        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        print('[Epoch {}] Training Time: {:.3f} Training Losses: {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if mAP > best_mAP:
            print("Saving parameters")
            net.export("{}/model".format(args.model_dir))
            best_mAP = mAP
            
        print("running mAP {}".format(mAP))
    
    print("best mAP {}".format(best_mAP))
        
# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = gluon.SymbolBlock.imports(
        '%s/model-symbol.json' % model_dir,
        ['data'],
        '%s/model-0000.params' % model_dir,
    )
    return net


def transform_fn(net, data,input_content_type,output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = np.array(parsed)
    
    x, image = gcv.data.transforms.presets.yolo.transform_test(mx.nd.array(nda),nda.shape[1])
    cid, score, bbox = net(x)    
    response_body = json.dumps(
        {"cid":cid.asnumpy().tolist()[0], 
         "score":score.asnumpy().tolist()[0], 
         "bbox": bbox.asnumpy().tolist()[0]
        })
    return response_body, output_content_type


def neo_preprocess(payload, content_type):

    parsed = json.loads(payload)
    nda = np.array(parsed)

    return nda

def neo_postprocess(result):

    response_body = json.dumps(result)
    content_type = 'application/json'

    return response_body, content_type