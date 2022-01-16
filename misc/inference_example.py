"""
some code taken from tutorial:
https://haochen23.github.io/2020/05/instance-segmentation-mask-rcnn.html#.Ydhsy3XML
"""


from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import time
import os

import cv2
import random
import warnings

# COCO class names as global var
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

PREDICTION_NAMES = [
    'person', 'car', 'bicycle', 'motorcyle', 'traffic light', 'truck', 'boat', 'bus',
     'potted plant', 'stop sign', 'fire hydrant'
]

def get_coloured_mask(mask):
    """
    a random colour is assigned to each detected/predicted object.
    great for working with pictures to determine each individual detected object
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],
                [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    
    return coloured_mask


def get_distinct_mask(mask, _class):
    """
    each class of interest (traffic related here - kinda) gets a unique colour
    """
    green = np.array([50, 205, 50])
    pink = np.array([199, 21, 133])
    yellow = np.array([255, 255, 102])
    blue = np.array([0, 0, 255])
    ocean = np.array([0, 191, 255])
    red = np.array([255, 0, 0])
    darkred = np.array([139, 0, 0])
    reddy = np.array([205, 92, 92])
    jungle = np.array([34, 139, 34])

    colour_dix = {'person': red, 'car': green, 'bicycle': yellow, 'motorcycle': yellow, 
                    'traffic light': pink, 'truck': blue, 'train': darkred, 'boat': ocean, 
                    'bus': reddy, 'potted plant': jungle, 'fire hydrant': red, 'stop sign': red}
    
    # prepare zero matrices for each colour channel in the size of the mask
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    # masks are boolean nd arrays, apply colours to Trues
    r[mask == 1], g[mask == 1], b[mask == 1] = colour_dix[_class]
    # 
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
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_scores.index(x) for x in pred_scores if x > confidence][-1]
    masks = (pred[0]['masks']>0.5).cpu().squeeze().detach().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_class[:pred_t+1]
    pred_scores = pred_scores[:pred_t+1]

    return masks, pred_boxes, pred_classes, pred_scores


def let_it_roll(path):
    """
    takes a list of image files and returns a video
    """
    pass


def segment_instance(model, input_path, output_path, confidence=0.9, rect_th=2, text_size=1, text_th=2, 
                        store=True, distinct=True):
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
    masks, boxes, pred_cls, pred_probs = get_prediction(model, input_path, confidence)
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        _class = pred_cls[i]
        mask = masks[i]
        box = boxes[i]
        pred_prob = round(pred_probs[i], 2)
        if _class in PREDICTION_NAMES:
            if distinct:
                rgb_mask = get_distinct_mask(mask, _class)
            else:
                rgb_mask = get_coloured_mask(mask)
            title = f'{_class},{str(pred_prob)}'
            img = cv2.addWeighted(img, 1, rgb_mask, 0.7, 0)
            cv2.rectangle(img, box[0], box[1],color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img=img, text=title, org=box[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=text_size, color=(255, 255, 255), thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    # plt.show()
    if store:
        plt.savefig(output_path)


def main():

    # time entire operation
    time1 = time.time()

    # supress warnings for cv2
    warnings.filterwarnings('ignore')

    # intitialize counter
    cnt = 0

    # load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # make it run on gpu
    model.cuda()

    # set to evaluation mode
    model.eval()

    output_files = []
    
    for file in os.listdir('input/'):
        filetype = file.split('.')[1]
        if filetype == 'jpg':
            in_path = f'input/{file}'
            out_path = f"output/{file.split('.')[0]}_pred.png"
            segment_instance(model, in_path, out_path)
            output_files.append(out_path)
            
            cnt += 1

    duration = round(time.time() - time1, 4)
    print(f'that took {duration} seconds for {cnt} image')

    let_it_roll(output_files)



if __name__ == '__main__':
    main()