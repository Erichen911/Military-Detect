#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import keras
from PyQt5 import QtCore, QtGui, QtWidgets
'''
def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    
    image_paths = []
    
    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]
        
    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)
        
        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
        
        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))         
'''

def non_max_suppression(x1,y1,x2,y2, probs, overlap_thresh=0.5, max_boxes=50):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(x1) == 0:
		return []

	# grab the coordinates of the bounding boxes
	#x1 = boxes[:, 0]
	#y1 = boxes[:, 1]
	#x2 = boxes[:, 2]
	#y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	#boxes = boxes[pick].astype("int")
	x1 = x1[pick]
	y1 = y1[pick]
	x2 = x2[pick]
	y2 = y2[pick]
	probs = probs[pick]
	return x1, y1, x2, y2, probs

def work(textedit,pic_label,input,model,output):
    input = input+"/"
    output = output+"/"
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file', default='config1.json')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam'
                           , default=input)
    argparser.add_argument('-o', '--output', default=output, help='path to output directory')
    args = argparser.parse_args()
    #_main_(args)  
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)
    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45
    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    #config['train']['saved_weights_name']
    keras.backend.clear_session()
    infer_model = load_model(config['train']['saved_weights_name'])
    ###############################
    #   Predict bounding boxes 
    ###############################  
    image_paths = []
    
    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]
        
    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    # the main loop
    strideN = 208 ##步长
    for image_path in image_paths:
        imageO = cv2.imread(image_path)
        print(image_path)
        image = QtGui.QPixmap(image_path)
        pic_label.setPixmap(image)
        pic_label.setScaledContents(True)
        
        (height,width,_) = imageO.shape
        mH = int((height-strideN)/strideN)
        mW = int((width-strideN)/strideN)
        
        ####对图像进行分割处理，网格搜索
        object_key = []
        object_pro = []
        object_x1 = []
        object_y1 = []
        object_x2 = []
        object_y2 = []
        
        for m in range(mH):
            for n in range (mW):
                print(m*mW+n)
                cursor = textedit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                cursor.insertText("Detecting: "+str(m*mW+n))
                cursor.insertText("\r\n")
                # textedit.append('Elapsed time = {}'.format(time.time() - st))
                textedit.setTextCursor(cursor)
                textedit.ensureCursorVisible()
                flag = False
                '''
                cursor = textedit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                cursor.insertText(str(m * mW + n))
                cursor.insertText("\r\n")
                '''
                imgCopy = imageO.copy()

                image = imgCopy[strideN*m:strideN*(m+2),strideN*n:strideN*(n+2)]##height,width
                
                # predict the bounding boxes
                boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
                # draw bounding boxes on the image using labels
                #draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
                # write the image with bounding boxes to file
                #aaa=image_path.split('/')
                #aab=aaa[-1].split('.')
                #cv2.imwrite(output_path + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1], np.uint8(image))
                
                ####存储所有的检测框
                labels = config['model']['labels']
                quiet=True
                key = []
                prob= []
                for box in boxes:
                    label_str = ''
                    label = -1
                    
                    for i in range(len(labels)):
                        if box.classes[i] > obj_thresh:
                            if label_str != '': label_str += ', '
                            label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')###概率值
                            key = labels[i]
                            prob= box.get_score()
                            label = i
                            if not quiet: print(label_str)
                    color = [0,0,255]
                    if key == "missile": color= [0,0,255]
                    if key == "oiltank": color= [0,159,255]
                    if key == "plane": color= [0,255,0]
                    if key == "warship": color= [255,0,0]

                    if label >= 0:
                        flag = True
                        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
                        width, height = text_size[0][0], text_size[0][1]
                        region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32') 
                        cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), color, thickness=2)
                        aaa=image_path.split('/')
                        aab=aaa[-1].split('.')
                        cv2.imwrite(output_path + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1], np.uint8(image))
                        print(output_path + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1])
                        #image = QtGui.QPixmap(output_path + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1])
                        #pic_label.setPixmap(image)
                        #pic_label.setScaledContents(True)

                        object_real_x1 = box.xmin + strideN*n
                        object_real_y1 = box.ymin + strideN*m
                        object_real_x2 = box.xmax + strideN*n
                        object_real_y2 = box.ymax + strideN*m
                    
                        object_key.append(key)
                        object_pro.append(prob)
                        object_x1.append(object_real_x1)
                        object_y1.append(object_real_y1)
                        object_x2.append(object_real_x2)
                        object_y2.append(object_real_y2)
                if flag:
                    flag = False
                    aaa = image_path.split('/')
                    aab = aaa[-1].split('.')
                    image = QtGui.QPixmap(output_path + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1])
                    pic_label.setPixmap(image)
                    pic_label.setScaledContents(True)

        ##非极大值抑制
        imgCopy2 = imageO.copy()
        object_name = ["missile","oiltank","plane","warship"]
        for object_class in range (len(object_name)):
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            prob=[]
            for numR in range (len(object_key)):
                if object_key[numR]==object_name[object_class]:
                    x1.append(object_x1[numR])
                    y1.append(object_y1[numR])
                    x2.append(object_x2[numR])
                    y2.append(object_y2[numR])
                    prob.append(object_pro[numR])
            if len(x1)>0:
                x1=np.array(x1)
                y1=np.array(y1)
                x2=np.array(x2)
                y2=np.array(y2)
                prob=np.array(prob)
            
                x1, y1, x2, y2, probs =  non_max_suppression(x1,y1,x2,y2, prob, overlap_thresh=0.5, max_boxes=30)     
            
                for numLR in range (len(x1)):
                    real_x1 = x1[numLR]
                    real_y1 = y1[numLR]
                    real_x2 = x2[numLR]
                    real_y2 = y2[numLR]
                
                    color = [0,0,255]
                    if object_name[object_class] == "missile": color= [0,0,255]
                    if object_name[object_class] == "oiltank": color = [0,159,255]
                    if object_name[object_class] == "plane": color = [0,255,0]
                    if object_name[object_class] == "warship": color = [255,0,0]
                    cv2.rectangle(imgCopy2,(real_x1, real_y1), (real_x2, real_y2), color,2)
        cv2.imwrite(output_path + image_path.split('/')[-1],imgCopy2)
        image = QtGui.QPixmap(output_path + image_path.split('/')[-1])
        pic_label.setPixmap(image)
        pic_label.setScaledContents(True)





