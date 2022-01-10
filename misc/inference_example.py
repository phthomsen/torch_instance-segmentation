"""
code taken from tutorial:
https://haochen23.github.io/2020/05/instance-segmentation-mask-rcnn.html#.Ydhsy3XML
"""


from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import time

import cv2
import random
import warnings


def get_coloured_mask(mask):
    """
    a random colour is assigned to each detected/predicted object
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],
                [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    
    return coloured_mask


def get_prediction(model, image_path, confidence):
    """
    image_path: path of image
    confidence: threshold for displaying a prediction

    - get image
    - convert to tensor
    - predict
    - masks, classes and bboxes are obtained from model
        - binary soft masks are applied on masks
    """
    # specify device of compute
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(image_path)
    transform = T.Compose([T.ToTensor()])
    # send input to cuda if possible
    img = transform(img).to(device)
    # after the prediction, image should be processed on cpu
    pred = model([img])
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
    masks = (pred[0]['masks']>0.5).cpu().squeeze().detach().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    return masks, pred_boxes, pred_class


def segment_instance(model, image_path, confidence=0.8, rect_th=2, text_size=2, text_th=2, store=True):
    """
    image_path: ...
    confidence: threshold for displaying a prediction
    rect_th: thickness of rectangle
    text_size: ...
    text_th: ...

    - get prediction
    - masks are given random colours
    - masks are added to the image with ration 1:0.8 with opencv
    """
    masks, boxes, pred_cls = get_prediction(model, image_path, confidence)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig('output/1.png')
        


def main():

    # time entire operation
    time1 = time.time()

    warnings.filterwarnings('ignore')

    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # make it run on gpu
    model.cuda()

    # set to evaluation mode
    model.eval()

    # COCO class names as global var
    global COCO_CLASS_NAMES
    COCO_CLASS_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    path = 'input/schnipsel_001.jpg'

    segment_instance(model, path)

    duration = round(time.time() - time1, 4)

    print(f'that took {duration} seconds for one image')

if __name__ == '__main__':
    main()