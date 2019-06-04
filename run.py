import keras_segmentation
import os

images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"

model = keras_segmentation.models.unet.vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train( 
    train_images =  images_path,
    train_annotations = segs_path,
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp= os.path.join(images_path, "001.Black_footed_Albatross", "Black_footed_Albatross_0001_796111.jpg"),
    out_fname="/tmp/out.png"
)


import matplotlib.pyplot as plt
plt.imshow(out)