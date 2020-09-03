import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score
from PyQt5 import QtCore, QtGui, QtWidgets



def get_map(pred, gt, f):
	T = {}
	P = {}
	fx, fy = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			if iou >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		if not gt_box['bbox_matched'] :#and not gt_box['difficult']
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	return T, P


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height, width, _) = img.shape

	if width <= height:
		f = img_min_side / width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side / height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width / float(new_width)
	fy = height / float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img, fx, fy

def compute_ap(recall, precision):
	""" Compute the average precision, given the recall and precision curves.
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.

	# Arguments
		recall:    The recall curve (list).
		precision: The precision curve (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], recall, [1.]))
	mpre = np.concatenate(([0.], precision, [0.]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap


def work(path, curelist, textedit):
	#print("I'm hearing too")
	#textedit.append("I'm hearing too")
	sys.setrecursionlimit(40000)
	test_path=path

	parser = OptionParser()

	parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default=test_path)
	parser.add_option("-n", "--num_rois", dest="num_rois",
					help="Number of ROIs per iteration. Higher means more memory use.", default=256)
	parser.add_option("--config_filename", dest="config_filename", help=
					"Location to read the metadata related to the training (generated when training).",
					default="config.pickle")
	parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
					default="pascal_voc"),#default="pascal_voc"

	(options, args) = parser.parse_args()

	if not options.test_path:   # if filename is not given
		parser.error('Error: path to test data must be specified. Pass --path to command line')


	if options.parser == 'pascal_voc':
		from keras_frcnn.pascal_voc_parser import get_data
	elif options.parser == 'simple':
		from keras_frcnn.simple_parser import get_data
	else:
		raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

	config_output_filename = options.config_filename

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	img_path = options.test_path




	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	#class_mapping = {v: k for k, v in class_mapping.iteritems()}
	class_mapping = {v: k for k, v in class_mapping.items()}
	print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = int(options.num_rois)

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (1024, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, 1024)


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

	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')


	#####
	all_imgs, _, _ = get_data(options.test_path)
	test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

	#test_imgs=test_imgs1[3111:4056]

	T = {}
	P = {}
	for idx, img_data in enumerate(test_imgs):
		print('{}/{}'.format(idx,len(test_imgs)))
		#textedit.append('{}/{}'.format(idx,len(test_imgs)))



		st = time.time()
		filepath = img_data['filepath']
		print(filepath)
		img = cv2.imread(filepath)

		X, fx, fy = format_img(img, C)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = model_rpn.predict(X)

		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.5)##0.7

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}

		for jk in range(R.shape[0] // C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0] // C.num_rois:
				# pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):

				if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		all_dets = []

		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.45)###0.5
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk, :]
				det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
				all_dets.append(det)


		print('Elapsed time = {}'.format(time.time() - st))
		cursor = textedit.textCursor()
		cursor.movePosition(QtGui.QTextCursor.End)
		cursor.insertText('{}/{}'.format(idx, len(test_imgs)))
		cursor.insertText("\r\n")
		cursor.insertText('Elapsed time = {}'.format(time.time() - st))
		cursor.insertText("\r\n")
		# textedit.append('Elapsed time = {}'.format(time.time() - st))
		textedit.setTextCursor(cursor)
		textedit.ensureCursorVisible()

		t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
		for key in t.keys():
			if key not in T:
				T[key] = []
				P[key] = []
			T[key].extend(t[key])
			P[key].extend(p[key])
	p1 = []
	t1 = []
	p1.append(P["airbase"])
	t1.append(T["airbase"])

	p1.append(P["harbour"])
	t1.append(T["harbour"])

	p1.append(P["island"])
	t1.append(T["island"])

	prefastr = []
	recfastr = []
	apfastr = []

	for m in range(len(p1)):
		p11 = np.zeros(len(p1[m]))
		for i in range(len(p1[m])):
			if p1[m][i] > 0.45:
				p11[i] = 1
			else:
				p11[i] = 0
		p_p1 = p11
		t_t1 = t1[m]

		false_positives = np.zeros((0,))
		true_positives = np.zeros((0,))
		scores = np.zeros((0,))
		num_annotations = 0.0
		nump = len(p_p1)

		# for n in range (1915):#(len(p_p1)):
		for n in range(len(p_p1)):
			if t_t1[n] == 1 and p_p1[n] == 1:
				true_positives = np.append(true_positives, 1)
				false_positives = np.append(false_positives, 0)
				scores = np.append(scores, p1[m][n])

			if t_t1[n] == 0 and p_p1[n] == 1:
				true_positives = np.append(true_positives, 0)
				false_positives = np.append(false_positives, 1)
				scores = np.append(scores, p1[m][n])

			# if t_t1[n]==1 and p_p1[n]==0:
			#   true_positives = np.append(true_positives, 0)
			#  false_positives = np.append(false_positives, 0)
			# scores = np.append(scores, p1[m][n])

			if t_t1[n] == 1:
				num_annotations = num_annotations + 1

		descending_indices = np.argsort(-scores)
		true_positives = true_positives[descending_indices]
		false_positives = false_positives[descending_indices]

		# compute false positives and true positives
		true_positives = np.cumsum(true_positives)
		false_positives = np.cumsum(false_positives)

		# compute recall and precision
		recall1 = true_positives / num_annotations
		precision1 = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

		average_precision = compute_ap(recall1, precision1)

		prefastr.append(precision1)
		recfastr.append(recall1)
		apfastr.append(average_precision)
	for i in range(0, len(recfastr)):
		curelist[i].setData(recfastr[i], prefastr[i])
	#fig = plt.figure(figsize=(18, 6))

'''
plt.subplot(131)
plt.plot(recfastr[0], prefastr[0], color='blue', linewidth=1)
plt.xlabel('recall', fontsize=14)
plt.ylabel('precision', fontsize=14)
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(apfastr[0]))

plt.subplot(132)
plt.plot(recfastr[1], prefastr[1], color='red', linewidth=1)
plt.xlabel('recall', fontsize=14)
plt.ylabel('precision', fontsize=14)
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(apfastr[1]))

plt.subplot(133)
plt.plot(recfastr[2], prefastr[2], color='green', linewidth=1)
plt.xlabel('recall', fontsize=14)
plt.ylabel('precision', fontsize=14)
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(apfastr[2]))
'''