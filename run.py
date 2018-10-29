import cv2
import numpy as np

from backprop import *


def normalize(x):
    x -= x.min()
    x = np.uint8(x * 255 / x.max())
    return x


def calc_gcam(x, lab, model, layer):
    grad_cam = GradCAM(model)
    gcam = grad_cam.execute(x, lab, layer)
    gcam = normalize(gcam)
    gcam = cv2.resize(gcam, (model.size, model.size))
    return gcam


def calc_gbp(x, lab, model, layer):
    guided_backprop = GuidedBackProp(model)
    gbp = guided_backprop.execute(x, lab, layer)
    return gbp

