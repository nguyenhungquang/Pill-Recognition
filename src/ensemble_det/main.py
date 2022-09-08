import os
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import boxes


from ensemble_boxes import weighted_boxes_fusion


image_folder = "VAIPE/public_test/pill/image"
boxes1_folder = "yolov5/runs/detect/v5s_832_tta/labels"
boxes2_folder = "yolov5/runs/detect/v5x6_1280_tta/labels"



if __name__ == '__main__':
    os.makedirs('ensemble_det/labels', exist_ok=True)
    for image_file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        name = image_file.split('.')[0]
        img = image.copy()

        # load boxes to fusion
        data1 = np.loadtxt(f"{boxes1_folder}/{name}.txt")
        data2 = np.loadtxt(f"{boxes2_folder}/{name}.txt")
        # data3 = np.loadtxt(f"{boxes3_folder}/{name}.txt")

        if data2.ndim == 1:
            data2 = np.expand_dims(data2, axis=0)
        if data1.ndim == 1:
            data1 = np.expand_dims(data1, axis=0)
        # if data3.ndim == 1:
        #     data3 = np.expand_dims(data3, axis=0)

        boxes1 = np.asarray(data1[:, 2:], dtype=np.int_)
        boxes2 = np.asarray(data2[:, 2:], dtype=np.int_)
        # boxes3 = np.asarray(data3[:, 2:], dtype=np.int_)

        try:
            class1 = np.asarray(data1[:, 0], dtype=np.int_)
        except IndexError:
            class1 = []
        try:
            class2 = np.asarray(data2[:, 0], dtype=np.int_)
        except IndexError:
            class2 = []
        # try:
        #     class3 = np.asarray(data3[:, 0], dtype=np.int_)
        # except IndexError:
        #     class3 = []

        try:
            conf1 = data1[:, 1]
        except IndexError:
            conf1 = []
        try:
            conf2 = data2[:, 1]
        except IndexError:
            conf2 = []
        # try:
        #     conf3 = data3[:, 1]
        # except IndexError:
        #     conf3 = []


        # fusion
        # hacky solution if box is an empty list/array
        det_boxes_list = [boxes1, boxes2]
        boxes_list = []
        for boxes_i in det_boxes_list:
            if boxes_i.size != 0:
                boxes_list.append(
                    boxes.normalize_box(image, boxes_i)
                )
        boxes_list = np.asarray(boxes_list)

        det_conf_list = [conf1, conf2]
        scores_list = []
        for conf_i in det_conf_list:
            if len(conf_i):
                scores_list.append(conf_i)
        scores_list = np.asarray(scores_list)

        det_label_list = [class1, class2]
        labels_list = []
        for label_i in det_label_list:
            if len(label_i):
                labels_list.append(label_i)

        # after fusion
        fboxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=0.4,
            conf_type='avg'
        )
        fnboxes = boxes.scale_box(image, fboxes)
        # save
        s = ""
        for i in range(len(fnboxes)): # each box in image
            c = int(labels[i])
            conf = scores[i]
            b = fnboxes[i]
            line = (c, conf, *b)
            with open(f"ensemble_det/labels/{name}.txt", 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
