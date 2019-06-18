import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

# Image_Name = [
#     "Bay_Breasted_Warbler_0042_797160", 
#     "Blue_Jay_0050_62974", 
#     "Caspian_Tern_0005_145929",
#     "Elegant_Tern_0053_150507",
#     "Rufous_Hummingbird_0067_59510",
#     "White_Breasted_Kingfisher_0065_73372",
#     "Florida_Jay_0044_64664",
#     "Pied_Billed_Grebe_0020_35958"
# ]

# origin_imgs = [
#     Image.open(os.path.join(test_images_path, img_name + '.jpg')) for img_name in Image_Name
# ]
# true_labels = [
#     Image.open(os.path.join("./Datasets/CUB_200_2011/true_labels", img_name + '.png')) for img_name in Image_Name
# ]
# result_imgs_fcn = [
#     Image.open(os.path.join("./Output_CUB/", "fcn_fcn_8_vgg" ,img_name + '.jpg')) for img_name in Image_Name
# ]
# result_imgs_pspnet = [
#     Image.open(os.path.join("./Output_CUB/", "pspnet_vgg_pspnet" ,img_name + '.jpg')) for img_name in Image_Name
# ]
# result_imgs_unet = [
#     Image.open(os.path.join("./Output_CUB/", "unet_vgg_unet" ,img_name + '.jpg')) for img_name in Image_Name
# ]
# result_imgs_segnet = [
#     Image.open(os.path.join("./Output_CUB/", "segnet_vgg_segnet" ,img_name + '.jpg')) for img_name in Image_Name
# ]
# # img = Image.open(os.path.join('images', '2007_000648' + '.jpg'))
# # gray = img.convert('L')
# # r,g,b = img.split()
# # img_merged = Image.merge('RGB', (r, g, b))


# plt.figure() #设置窗口大小
# plt.suptitle('Image Segmentation in CUB-200 Dataset') # 图片名称
# for i in range(8):
#     plt.subplot(6,8,i+1)#, plt.title('original image')
#     plt.imshow(origin_imgs[i]), plt.axis('off')
# for i in range(8):
#     plt.subplot(6,8,8+i+1)#, plt.title('true label')
#     plt.imshow(true_labels[i]), plt.axis('off')
# for i in range(8):
#     plt.subplot(6,8,16+i+1)#, plt.title('FCN')
#     plt.imshow(result_imgs_fcn[i]), plt.axis('off')
# for i in range(8):
#     plt.subplot(6,8,24+i+1)#, plt.title('PSPNet')
#     plt.imshow(result_imgs_pspnet[i]), plt.axis('off')
# for i in range(8):
#     plt.subplot(6,8,32+i+1)#, plt.title('U-Net')
#     plt.imshow(result_imgs_unet[i]), plt.axis('off')
# for i in range(8):
#     plt.subplot(6,8,40+i+1)#, plt.title('SegNet')
#     plt.imshow(result_imgs_segnet[i]), plt.axis('off')

# plt.show()


image_names = ["1.jpg", "2.jpg", "5.jpg", "2011_002114.jpg"]
images = [
    Image.open(os.path.join("./Output_VOC", img_name)) for img_name in image_names
]
plt.figure() #设置窗口大小
# plt.suptitle() # 图片名称
plt.subplot(1,4,1), plt.title('epoch 10')
plt.imshow(images[0]), plt.axis('off')
plt.subplot(1,4,2), plt.title('epoch 20')
plt.imshow(images[1]), plt.axis('off')
plt.subplot(1,4,3), plt.title('epoch 50')
plt.imshow(images[2]), plt.axis('off')
plt.subplot(1,4,4), plt.title('ecpoch 200')
plt.imshow(images[3]), plt.axis('off')
plt.show()

