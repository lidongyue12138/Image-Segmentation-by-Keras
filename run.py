import keras_segmentation
from keras.models import load_model
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2, 3" 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

DATA_NAME = "VOC"
EPOCH = [5, 10]
CHECKOUTPOINT_PATH = "./tmp/voc_5_10"

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
model = keras_segmentation.models.unet.resnet50_unet(n_classes=class_num,  input_height=416, input_width=608)
# model.load_weights("./tmp/cub_psspnet_vgg_pspnet.9")

for i in range(EPOCH[0]):

    '''
    Train
    '''
    model.train( 
        train_images =  train_images_path,
        train_annotations = train_segs_path,
        checkpoints_path = CHECKOUTPOINT_PATH , epochs=EPOCH[1], verify_dataset = False
    )

    '''
    Output
    '''
    # for (i, image_dir) in enumerate(os.listdir(test_images_path)):
    #     if i%50 == 0:
    #         out = model.predict_segmentation(
    #             inp= os.path.join(test_images_path, image_dir),
    #             out_fname= os.path.join("./Output/", image_dir)
    #         )
    #         # import matplotlib.pyplot as plt
    #         # plt.imshow(out)

    '''
    Test mIoU
    '''
    test_image_list = [os.path.join(test_images_path, i) for i in os.listdir(test_images_path)]
    test_segs_list = [os.path.join(test_segs_path, i) for i in os.listdir(test_segs_path)]

    model.evaluate_segmentation(
        inp_images = test_image_list,
        annotations = test_segs_list
    )