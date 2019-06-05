import cv2
import os

images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"
save_path = "./Datasets/CUB_200_2011/converted/"

def convert_CUB(image):
    image[image[:, :, 0]!=0] = 1
    return image

if __name__ == "__main__":
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for class_dir in os.listdir(segs_path):
        for image_dir in os.listdir(os.path.join(segs_path, class_dir)):
            img = cv2.imread(os.path.join(segs_path, class_dir, image_dir), 1)
            img = convert_CUB(img)
            if not os.path.isdir(os.path.join(save_path, class_dir)):
                os.makedirs(os.path.join(save_path, class_dir))
            cv2.imwrite(os.path.join(save_path, class_dir, image_dir),img)
