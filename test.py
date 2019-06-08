import keras_segmentation
import numpy as np
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" 

DATA_NAME = "CUB"

if DATA_NAME == "VOC":
    train_images_path = "./Datasets/VOC/train/imgs/"
    train_segs_path = "./Datasets/VOC/train/segs/"
    test_images_path = "./Datasets/VOC/test/imgs/"
    test_segs_path = "./Datasets/VOC/test/segs"
    class_num = 21
if DATA_NAME == "CUB":
    train_images_path = "./Datasets/CUB_200_2011/train/imgs/"
    train_segs_path = "./Datasets/CUB_200_2011/train/segs/"
    test_images_path = "./Datasets/CUB_200_2011/test/imgs/"
    test_segs_path = "./Datasets/CUB_200_2011/test/segs"
    class_num = 2
'''
Change model name
'''
model = keras_segmentation.models.pspnet.vgg_segnet(n_classes=class_num,  input_height=192, input_width=192)
model.load_weights("./tmp/cub_psspnet_vgg_pspnet.9")

# load any of the 3 pretrained models

for (i, image_dir) in enumerate(os.listdir(test_images_path)):
    if i%50 == 0:
        out = model.predict_segmentation(
            inp= os.path.join(test_images_path, image_dir),
            out_fname= os.path.join("./Output_CUB/", image_dir))
