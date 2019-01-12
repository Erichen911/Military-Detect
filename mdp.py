import torch
from torch.autograd import Variable

import numpy as np

from models import FeatureExtractor

class MDP:
    def __init__(self, device, n_out, alpha, tau, eta, hist_size):
        self.device = device
        self.extractor = FeatureExtractor().to(device)
        self.n_out = n_out
        self.alpha = alpha
        self.tau = tau
        self.eta = eta
        self.hist_size = hist_size
        self.actions = None

        self.image = None
        self.ground_box = None
        self.running_box = None

    def step(self, A):
        assert self.image is not None and \
               self.ground_box is not None and \
               self.running_box is not None

        x1, y1, x2, y2 = self.running_box
        self.actions.append(A)
        del self.actions[0]

        alpha_w = int(self.alpha * (x2 - x1))
        alpha_h = int(self.alpha * (y2 - y1))

        done = False
        if A == 0:
            # right
            x1 = x1 + alpha_w
        elif A == 1:
            # left
            x2 = x2 - alpha_w
        elif A == 2:
            # up
            y2 = y2 = y2 - alpha_h
        elif A == 3:
            # down
            y1 = y1 + alpha_h
        elif A == 4:
            # bigger
            x1 = x1 - alpha_w
            y1 = y1 - alpha_h
            x2 = x2 + alpha_w
            y2 = y2 + alpha_h
        elif A == 5:
            # smaller
            x1 = x1 + alpha_w
            y1 = y1 + alpha_h
            x2 = x2 - alpha_w
            y2 = y2 - alpha_h
        elif A == 6:
            # fatter
            y1 = y1 + alpha_h
            y2 = y2 - alpha_h
        elif A == 7:
            # taller
            x1 = x1 + alpha_w
            x2 = x2 - alpha_w
        elif A == 8:
            # stop trigger
            done = True

        predicted_box = self._correct_bounds([x1, y1, x2, y2])
        if done:
            # special reward when stop is trigerred
            iou = self._compute_iou(predicted_box, self.ground_box)
            R = self.eta if iou >= self.tau else -1 * self.eta
            # return initial S since this next state will anyway be discarded
            return self._build_state(self.image), R, done

        S_pr, R = self._transition(predicted_box)

        # update the running bounding box
        self.running_box = predicted_box

        return S_pr, R, done

    def _correct_bounds(self, box):
        new_box = []
        for c in box:
            c_new = np.clip(c, 0, 224)
            new_box.append(int(c_new))
        return new_box

    def _build_state(self, image):
        with torch.no_grad():
            image = Variable(torch.FloatTensor(image)).to(self.device)
            features = self.extractor(image).cpu()
        a_hist = self._get_action_history()

        # final state is concat of image feature & action history
        S = torch.cat([features, a_hist], dim=1)
        return S

    def _transition(self, predicted_box):
        focused_image = torch.zeros(self.image.size())
        x1, y1, x2, y2 = predicted_box
        focused_image[:, :, x1:x2, y1:y2] = self.image[:, :, x1:x2, y1:y2]

        S_pr = self._build_state(focused_image)

        old_iou = self._compute_iou(self.running_box, self.ground_box)
        new_iou = self._compute_iou(predicted_box, self.ground_box)

        delta = new_iou - old_iou
        R = -1 if delta < 0 else 1

        return S_pr, R

    def _compute_iou(self, box1, box2):
        '''
        Computes the intersection over union measure for boxes
        box1 and box2.

        Taken from:
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        '''
        # determine the (x, y)-coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(box1Area + box2Area - interArea)

        # return the intersection over union value
        return iou

    def _get_action_history(self):
        # ensure history length
        assert len(self.actions) == self.hist_size
        a_hist = torch.zeros(self.hist_size, self.n_out)

        # build 1-hot vector
        for i, a in enumerate(self.actions):
            if a:
                a_hist[i][a] = 1
        return a_hist.view(1, -1)

    def reset(self, image, ground_box):
        # initialize action history
        self.actions = [None] * self.hist_size

        # initialize running states
        self.image = image
        self.ground_box = ground_box
        self.running_box = [0, 0, 224, 224]

        return self._build_state(image)
