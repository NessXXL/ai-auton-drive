
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import colorsys


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        return 0 if x4 < x1 else min(x2, x4) - x1
    else:
        return 0 if x2 < x3 else min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def preprocess_input(image_pil, net_h, net_w):
    image = np.asarray(image_pil)
    new_h, new_w, _ = image.shape
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h
    new_w, new_h = int(new_w), int(new_h)
    resized = cv2.resize(image / 255., (new_w, new_h))
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
              int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    return np.expand_dims(new_image, 0)


def decode_netout(netout_, obj_thresh, anchors_, image_h, image_w, net_h, net_w):
    netout_all = deepcopy(netout_)
    boxes_all = []
    for i in range(len(netout_all)):
        netout = netout_all[i][0]
        anchors = anchors_[i]
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh
        for i in range(grid_h * grid_w):
            row = i // grid_w
            col = i % grid_w
            for b in range(nb_box):
                objectness = netout[row][col][b][4]
                classes = netout[row][col][b][5:]
                if (classes <= obj_thresh).all():
                    continue
                x, y, w, h = netout[row][col][b][:4]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[b][0] * np.exp(w) / net_w
                h = anchors[b][1] * np.exp(h) / net_h
                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)
        boxes_all += boxes
    boxes_all = correct_yolo_boxes(boxes_all, image_h, image_w, net_h, net_w)
    return boxes_all


def correct_yolo_boxes(boxes_, image_h, image_w, net_h, net_w):
    boxes = deepcopy(boxes_)
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h
    for i in range(len(boxes)):
        x_offset = (net_w - new_w) / 2. / net_w
        x_scale = float(new_w) / net_w
        y_offset = (net_h - new_h) / 2. / net_h
        y_scale = float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    return boxes


def do_nms(boxes_, nms_thresh, obj_thresh):
    boxes = deepcopy(boxes_)
    if len(boxes) > 0:
        num_class = len(boxes[0].classes)
    else:
        return []
    for c in range(num_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
    new_boxes = []
    for box in boxes:
        for i in range(num_class):
            if box.classes[i] > obj_thresh:
                box.label = i
                box.score = box.classes[i]
                new_boxes.append(box)
                break
    return new_boxes


def draw_boxes(image_, boxes, labels):
    image = image_.copy()
    image_w, image_h = image.size
    try:
        font = ImageFont.truetype(
            font='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
            size=np.floor(3e-2 * image_h + 0.5).astype('int32')
        )
    except Exception:
        font = ImageFont.load_default()
    thickness = (image_w + image_h) // 300
    hsv_tuples = [(x / len(labels), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    for i, box in reversed(list(enumerate(boxes))):
        c = box.get_label()
        predicted_class = labels[c]
        score = box.get_score()
        top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textbbox((0, 0), label, font)
        label_size = (label_size[2], label_size[3])
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_w, np.floor(right + 0.5).astype('int32'))
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        if right > left and bottom > top:
            draw.rectangle([left, top, right, bottom], outline=colors[c], width=thickness)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image


def detect_image(image_pil, model, anchors, labels, obj_thresh=0.4, nms_thresh=0.45, net_h=416, net_w=416):
    image_w, image_h = image_pil.size
    new_image = preprocess_input(image_pil, net_h, net_w)
    yolo_outputs = model.predict(new_image)
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    return draw_boxes(image_pil, boxes, labels)
