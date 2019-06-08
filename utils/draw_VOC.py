import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_NAME = "VOC"

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

Image_Name = [
    "2011_002956",
    "2011_002114",
    "2010_002962",
    "2007_003205",
    "2009_002844",
    "2010_001184",
    "2011_002135",
    "2010_005800"
]

origin_imgs = [
    Image.open(os.path.join(test_images_path, img_name + '.jpg')) for img_name in Image_Name
]
true_labels = [
    Image.open(os.path.join("./Datasets/VOC/SegmentationClass", img_name + '.png')) for img_name in Image_Name
]
result_imgs_fcn = [
    Image.open(os.path.join("./Output_VOC/", "fcn_8_vgg_1" ,img_name + '.jpg')) for img_name in Image_Name
]
result_imgs_pspnet = [
    Image.open(os.path.join("./Output_VOC/", "pspnet_vgg_pspnet_1" ,img_name + '.jpg')) for img_name in Image_Name
]
result_imgs_unet = [
    Image.open(os.path.join("./Output_VOC/", "unet_resnet_50_2" ,img_name + '.jpg')) for img_name in Image_Name
]
result_imgs_segnet = [
    Image.open(os.path.join("./Output_VOC/", "segnet_resnet_50_1" ,img_name + '.jpg')) for img_name in Image_Name
]
# img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))
# gray = img.convert('L')
# r,g,b = img.split()
# img_merged = Image.merge('RGB', (r, g, b))


plt.figure() #设置窗口大小
plt.suptitle('Image Segmentation in VOC-2012 Dataset') # 图片名称
for i in range(8):
    plt.subplot(6,8,i+1)#, plt.title('original image')
    plt.imshow(origin_imgs[i]), plt.axis('off')
for i in range(8):
    plt.subplot(6,8,8+i+1)#, plt.title('true label')
    plt.imshow(true_labels[i]), plt.axis('off')
for i in range(8):
    plt.subplot(6,8,16+i+1)#, plt.title('FCN')
    plt.imshow(result_imgs_fcn[i]), plt.axis('off')
for i in range(8):
    plt.subplot(6,8,24+i+1)#, plt.title('PSPNet')
    plt.imshow(result_imgs_pspnet[i]), plt.axis('off')
for i in range(8):
    plt.subplot(6,8,32+i+1)#, plt.title('U-Net')
    plt.imshow(result_imgs_unet[i]), plt.axis('off')
for i in range(8):
    plt.subplot(6,8,40+i+1)#, plt.title('SegNet')
    plt.imshow(result_imgs_segnet[i]), plt.axis('off')

plt.show()
