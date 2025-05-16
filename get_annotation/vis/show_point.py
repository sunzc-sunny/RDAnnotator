import numpy as np
import cv2
import os
import random

import matplotlib.pyplot as plt


def visualize_anchors(image, anchors, color, save_path=None):

    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image

def visualize_anchors2(image, anchors, color, save_path=None):

    for anchor in anchors:
        x1, y1, w, h = anchor
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 1)


    return image
