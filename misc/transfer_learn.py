"""
goal is to take the pre trained mask rcnn model and train it on comma10k data, following
this article: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

use functions from https://github.com/pytorch/vision.git

model is trained on the coco data set, which has 80 + 1(background) classes. comma10k has 5:

1 - #402020 - road (all parts, anywhere nobody would look at you funny for driving) - (64, 32, 32)
2 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks) (255, 0, 0)
3 - #808060 - undrivable (128, 128, 96)
4 - #00ff66 - movable (vehicles and people/animals) (0, 255, 102)
5 - #cc00ff - my car (and anything inside it, including wires, mounts, etc. No reflections) (204, 0, 255)
"""

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import warnings

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')


class CommaDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "commaImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "commaMasks"))))
        self.color_dic = {"road": [64, 32, 32], "lane markings": [255, 0, 0], "undrivable": [128, 128, 96], 
                            "movable": [0, 255, 102], "my car": [204, 0, 255]}

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "commaImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "commaMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)

        mask = np.array(mask)
        # objects are encoded as different colors
        rgb_classes = self.color_dic.values()
        num_classes = len(rgb_classes)
        
        # masks should have dimensions (num_classes, img.shape[0], img.shape[1])
        masks = np.array([np.all(mask == rgb, axis=-1) for rgb in rgb_classes])

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        target = {}
        target["masks"] = masks
        target["image_id"] = image_id
        # the helper methods for rcnn models require boxes
        target['boxes'] = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def main():
    # train on gpu if possible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 10

    # load the model and send it to our device
    pretrained_model = get_instance_segmentation_model(5)
    pretrained_model.to(device)

    # define the dataset
    dataset_train = CommaDataset('commaData', get_transform(train=True))
    dataset_validate = CommaDataset('commaData', get_transform(train=False))

    # split the data
    torch.manual_seed(69)
    inds = torch.randperm(len(dataset_train)).tolist()
    train_split = int(0.9 * len(inds))
    train_inds = inds[:train_split]
    val_inds = inds[train_split:]
    dataset_train = torch.utils.data.Subset(dataset_train, train_inds)
    dataset_validate = torch.utils.data.Subset(dataset_validate, val_inds)

    # now get data laoded
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=2,
                                               shuffle=True, num_workers=16, 
                                               collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_validate, batch_size=1, 
                                             shuffle=False, num_workers=16, 
                                             collate_fn=utils.collate_fn)
    
    params = [p for p in pretrained_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        train_one_epoch(pretrained_model, optimizer=optimizer, data_loader=train_loader, 
                        device=device, epoch=epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(pretrained_model, val_loader, device=device)


if __name__ == "__main__":
    main()
    