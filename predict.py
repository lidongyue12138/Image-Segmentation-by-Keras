import keras_segmentation
from keras.models import load_model
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 

# train_images_path = "./Datasets/CUB_200_2011/train/imgs/"
# train_segs_path = "./Datasets/CUB_200_2011/train/segs/"
# test_images_path = "./Datasets/CUB_200_2011/test/imgs/"
# test_segs_path = "./Datasets/CUB_200_2011/test/segs"

train_images_path = "./Datasets/VOC/train/imgs/"
train_segs_path = "./Datasets/VOC/train/segs/"
test_images_path = "./Datasets/VOC/test/imgs/"
test_segs_path = "./Datasets/VOC/test/segs"

model = keras_segmentation.models.unet.vgg_unet(n_classes=21 ,  input_height=416, input_width=608)
model.load_weights("./tmp/voc_fcn_fcn_32_vgg.4")

'''
Output
'''
for (i, image_dir) in enumerate(os.listdir(test_images_path)):
    if i%50 == 0:
        out = model.predict_segmentation(
            inp= os.path.join(test_images_path, image_dir),
            out_fname= os.path.join("./Output_VOC/", image_dir)
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(out)
    