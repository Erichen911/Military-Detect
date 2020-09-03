#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation
from generator import BatchGenerator
from utils.utils import normalize, evaluate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from yolo import create_yolov3_model, dummy_loss
import time

def _main_(args,textedit):
    config_path = args.conf
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Create the validation generator
    ###############################  
    valid_ints, labels = parse_voc_annotation(
        config['valid']['valid_annot_folder'], 
        config['valid']['valid_image_folder'], 
        config['valid']['cache_name'],
        config['model']['labels']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    #labels = sorted(labels)
   
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 0,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    
        ###############################
    #   Create the model 
    ##############################
    
    train_model, infer_model = create_yolov3_model(
            nb_class            = len(labels), 
            anchors             = config['model']['anchors'],  
            max_box_per_image   = 0, 
            max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
            batch_size          = config['train']['batch_size'], 
            warmup_batches      = 0,
            ignore_thresh       = config['train']['ignore_thresh'],
            grid_scales         = config['train']['grid_scales'],
            obj_scale           = config['train']['obj_scale'],
            noobj_scale         = config['train']['noobj_scale'],
            xywh_scale          = config['train']['xywh_scale'],
            class_scale         = config['train']['class_scale'],
        )  
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    
    saved_weights_name  = config['train']['saved_weights_name']
    lr                  = config['train']['learning_rate'],
    
    infer_model.load_weights(saved_weights_name)
    optimizer = Adam(lr=lr, clipnorm=0.001)
    infer_model.compile(loss=dummy_loss, optimizer=optimizer) 
    
    infer_model.summary()
    #infer_model = load_model(config['train']['saved_weights_name'])
    print("Begin to compute")
    # compute mAP for all the classes
    recall, precision, average_precisions = evaluate(infer_model, valid_generator, textedit)
    print("End compute")
    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
        textedit.append(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    textedit.append('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
    textedit.append("Finish the project")
    return recall, precision, average_precisions
        

def work(curelist,textedit,configfile):
    if os.path.exists("./yolo3_valid.pkl"):
        os.remove("./yolo3_valid.pkl")
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file', default=configfile)
    args = argparser.parse_args()


    textedit.append("Detecting")
    recalls, precisions, average_precisions = _main_(args,textedit)
    for i in range(0,len(curelist)):
        curelist[i].setData(recalls[i], precisions[i])
