import keras_segmentation
from keras.models import load_model
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 

train_images_path = "./Datasets/CUB_200_2011/train/imgs/"
train_segs_path = "./Datasets/CUB_200_2011/train/segs/"
test_images_path = "./Datasets/CUB_200_2011/test/imgs/"
test_segs_path = "./Datasets/CUB_200_2011/test/segs"

model = keras_segmentation.models.fcn.fcn_8(n_classes=2 ,  input_height=416, input_width=608  )
# model.load_weights("./tmp/vgg_unet_1.1")

model.train( 
    train_images =  train_images_path,
    train_annotations = train_segs_path,
    checkpoints_path = "./tmp/fcn_fcn8_1" , epochs=2
)

# out = model.predict_segmentation(
#     inp= os.path.join(images_path, "001.Black_footed_Albatross", "Black_Footed_Albatross_0001_796111.jpg"),
#     out_fname="./tmp/out.png"
# )

test_image_list = [os.path.join(test_images_path, i) for i in os.listdir(test_images_path)]
test_segs_list = [os.path.join(test_segs_path, i) for i in os.listdir(test_segs_path)]
# import matplotlib.pyplot as plt
# plt.imshow(out)

model.evaluate_segmentation(
    inp_images = test_image_list,
    annotations = test_segs_list
)