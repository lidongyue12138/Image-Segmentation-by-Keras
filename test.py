import cv2
import os

images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"

class_dir = os.listdir(segs_path)[0]
image_dir = os.listdir(os.path.join(segs_path, class_dir))[0]

img = cv2.imread(os.path.join(segs_path, class_dir, image_dir), 0)

# img[img[:, :, 0]==255] = 1
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()