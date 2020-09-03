from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from PyQt5 import QtCore, QtGui, QtWidgets
import keras

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def non_max_suppression(x1, y1, x2, y2, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(x1) == 0:
        return []

    # grab the coordinates of the bounding boxes
    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]

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
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    # boxes = boxes[pick].astype("int")
    x1 = x1[pick]
    y1 = y1[pick]
    x2 = x2[pick]
    y2 = y2[pick]
    probs = probs[pick]
    return x1, y1, x2, y2, probs


def work(textedit,pic_label,input,model,output):
    #run test_frcnn.py -p ./testImages/
    sys.setrecursionlimit(40000)
    keras.backend.clear_session()
    test_path=input + "/"
    output = output + "/"


    parser = OptionParser()

    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default=test_path)
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                    help="Number of ROIs per iteration. Higher means more memory use.", default=256)
    parser.add_option("--config_filename", dest="config_filename", help=
                    "Location to re ad the metadata related to the training (generated when training).",
                    default="config.pickle")
    parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

    (options, args) = parser.parse_args()

    if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')


    config_output_filename = options.config_filename

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    if C.network == 'resnet50':
        import keras_frcnn.resnet as nn
    elif C.network == 'vgg':
        import keras_frcnn.vgg as nn

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    img_path = options.test_path

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs = []

    classes = {}

    bbox_threshold = 0.8

    visualise = True

    strideN = 400 ##步长
    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        print("开始检测：")
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)

        filepath = os.path.join(img_path,img_name)

        image = QtGui.QPixmap(filepath)
        pic_label.setPixmap(image)
        pic_label.setScaledContents(True)

        imgO = cv2.imread(filepath)
        (height,width,_) = imgO.shape
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

                imgCopy = imgO.copy()

                img = imgCopy[strideN*m:strideN*(m+2),strideN*n:strideN*(n+2)]##height,width

                st = time.time()

                X, ratio = format_img(img, C)
                if K.image_dim_ordering() == 'tf':
                    X = np.transpose(X, (0, 2, 3, 1))

                # get the feature maps and output from the RPN
                [Y1, Y2, F] = model_rpn.predict(X)

                R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

                # convert from (x1,y1,x2,y2) to (x,y,w,h)
                R[:, 2] -= R[:, 0]
                R[:, 3] -= R[:, 1]

                # apply the spatial pyramid pooling to the proposed regions
                bboxes = {}
                probs = {}

                for jk in range(R.shape[0]//C.num_rois + 1):
                    ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                    if ROIs.shape[1] == 0:
                        break

                    if jk == R.shape[0]//C.num_rois:
                        #pad R
                        curr_shape = ROIs.shape
                        target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                        ROIs_padded[:, :curr_shape[1], :] = ROIs
                        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                        ROIs = ROIs_padded

                    [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                    for ii in range(P_cls.shape[1]):
                        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                            continue
                        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                        if cls_name not in bboxes:
                            bboxes[cls_name] = []
                            probs[cls_name] = []

                        (x, y, w, h) = ROIs[0, ii, :]
                        cls_num = np.argmax(P_cls[0, ii, :])

                        try:
                            (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                            tx /= C.classifier_regr_std[0]
                            ty /= C.classifier_regr_std[1]
                            tw /= C.classifier_regr_std[2]
                            th /= C.classifier_regr_std[3]
                            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                        except:
                            pass
                        bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                        probs[cls_name].append(np.max(P_cls[0, ii, :]))

                all_dets = []

                for key in bboxes:
                    print(key)
                    bbox = np.array(bboxes[key])

                    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                    for jk in range(new_boxes.shape[0]):
                        print("test")
                        (x1, y1, x2, y2) = new_boxes[jk,:]
                        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                        color = [0,0,255]
                        if key == "airbase": color= [0,0,255]
                        if key == "harbour": color = [0,159,255]
                        if key == "island": color = [0,255,0]


                        print(real_x1)
                        print(real_y1)
                        print(real_x2)
                        print(real_y2)

                        cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), color,2)

                        textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                        print(textLabel)
                        all_dets.append((key,100*new_probs[jk]))

                        object_real_x1 = real_x1 + strideN*n
                        object_real_y1 = real_y1 + strideN*m
                        object_real_x2 = real_x2 + strideN*n
                        object_real_y2 = real_y2 + strideN*m

                        object_key.append(key)
                        object_pro.append(new_probs[jk])
                        object_x1.append(object_real_x1)
                        object_y1.append(object_real_y1)
                        object_x2.append(object_real_x2)
                        object_y2.append(object_real_y2)

                        #(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                        #textOrg = (real_x1, real_y1-0)

                        #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                        #cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                        #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

                print('Elapsed time = {}'.format(time.time() - st))
                cursor = textedit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                cursor.insertText(str(m*mW+n))
                cursor.insertText("\r\n")
                cursor.insertText('Elapsed time = {}'.format(time.time() - st))
                cursor.insertText("\r\n")
                # textedit.append('Elapsed time = {}'.format(time.time() - st))
                textedit.setTextCursor(cursor)
                textedit.ensureCursorVisible()
                #print(all_dets)
                aaa=filepath.split('/')
                aab=aaa[-1].split('.')
                cv2.imwrite(output + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1],img)
                image = QtGui.QPixmap(output + aab[0]  + '_' + str(m*mW+n+1) + '.' + aab[1])
                pic_label.setPixmap(image)
                pic_label.setScaledContents(True)
                #cv2.imwrite('./results_imgs/{}.jpg'.format(m*mW+n),img)

        ##非极大值抑制
        imgCopy2 = imgO.copy()
        object_name = ["airbase","harbour","island"]
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

                x1, y1, x2, y2, probs =  non_max_suppression(x1,y1,x2,y2, prob, overlap_thresh=0.5, max_boxes=300)

                for numLR in range (len(x1)):
                    real_x1 = x1[numLR]
                    real_y1 = y1[numLR]
                    real_x2 = x2[numLR]
                    real_y2 = y2[numLR]

                    color = [0,0,255]
                    if object_name[object_class] == "airbase": color= [0,0,255]
                    if object_name[object_class] == "harbour": color = [0,159,255]
                    if object_name[object_class] == "island": color = [0,255,0]
                    cv2.rectangle(imgCopy2,(real_x1, real_y1), (real_x2, real_y2), color,2)
        #cv2.imwrite('./results_imgs/{}.jpg'.format(9999),imgCopy2)
        cv2.imwrite(output + filepath.split('/')[-1],imgCopy2)
        image = QtGui.QPixmap(output + filepath.split('/')[-1])
        pic_label.setPixmap(image)
        pic_label.setScaledContents(True)

        '''
        for numR in range (len(object_key)):
            key = object_key[numR]
            color = [0,0,255]
            if key == "airbase": color= [0,0,255]
            if key == "harbour": color = [0,159,255]
            if key == "island": color = [0,255,0]
            
            cv2.rectangle(imgO,(object_x1[numR], object_y1[numR]), (object_x2[numR], object_y2[numR]), color,2)
        cv2.imwrite('./results_imgs/{}.jpg'.format(100),imgO)
        '''

