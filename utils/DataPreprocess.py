import cv2
import os
import numpy as np

images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"
save_path = "./Datasets/CUB_200_2011/converted/"

CUB_DATA_PATH = "./Datasets/CUB_200_2011/"
IMAGE_LIST_DIR = "./Datasets/CUB_200_2011/images.txt"
TRAIN_TEST_DIR = "./Datasets/CUB_200_2011/train_test_split.txt"

VOC_DATA_PATH = "./Datasets/VOC/"
VOC_TRAIN_TXT = "./Datasets/VOC/ImageSets/Segmentation/train.txt"
VOC_VAL_TXT = "./Datasets/VOC/ImageSets/Segmentation/train.txt"

def convert_CUB(image):
    image[image[:, :, 0]!=0] = 1
    return image

def convert_segmentations_CUB():
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for class_dir in os.listdir(segs_path):
        for image_dir in os.listdir(os.path.join(segs_path, class_dir)):
            img = cv2.imread(os.path.join(segs_path, class_dir, image_dir), 1)
            img = convert_CUB(img)
            if not os.path.isdir(os.path.join(save_path, class_dir)):
                os.makedirs(os.path.join(save_path, class_dir))
            cv2.imwrite(os.path.join(save_path, class_dir, image_dir),img)

'''
Read train & test images and save them to corresponding dir
'''
def make_dataset_CUB():
    '''
    Make necessary dirs
    '''
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "train")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "train"))
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "test")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "test"))
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "train", "imgs")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "train", "imgs"))
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "train", "segs")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "train", "segs"))
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "test", "imgs")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "test", "imgs"))
    if not os.path.exists(os.path.join(CUB_DATA_PATH, "test", "segs")):
        os.makedirs(os.path.join(CUB_DATA_PATH, "test", "segs"))
    ''' End '''
    image_list = open(IMAGE_LIST_DIR, "r").readlines()
    train_test_list  = open(TRAIN_TEST_DIR, "r").readlines()
    for line, trian_test in zip(image_list, train_test_list):
        num, image_dir = line.split()
        ifTrain = int(trian_test.split()[1])
        
        seg_dir = image_dir[:-4] + ".png"

        image_name = os.path.basename(image_dir)
        seg_name = image_name.split(".")[0] + ".png"
        if ifTrain:
            # Save images
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "images", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "train", "imgs", image_name), img)
            # Save segmentations
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "converted", seg_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "train", "segs", seg_name), img)
        else:
            # Save images
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "images", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "test", "imgs", image_name), img)
            # Save segmentations
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "converted", seg_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "test", "segs", seg_name), img)

def convert_VOC(img):
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==0)] = 0
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==128)] = 1
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==0)] = 2
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==128)] = 3
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==0)] = 4
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==128)] = 5
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==0)] = 6
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==128)] = 7
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==64)] = 8
    img[(img[:,:,0]==0) * (img[:, :, 1]==0) * (img[:,:,2]==192)] = 9
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==64)] = 10
    img[(img[:,:,0]==0) * (img[:, :, 1]==128) * (img[:,:,2]==192)] = 11
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==64)] = 12
    img[(img[:,:,0]==128) * (img[:, :, 1]==0) * (img[:,:,2]==192)] = 13
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==64)] = 14
    img[(img[:,:,0]==128) * (img[:, :, 1]==128) * (img[:,:,2]==192)] = 15
    img[(img[:,:,0]==0) * (img[:, :, 1]==64) * (img[:,:,2]==0)] = 16
    img[(img[:,:,0]==0) * (img[:, :, 1]==64) * (img[:,:,2]==128)] = 17
    img[(img[:,:,0]==0) * (img[:, :, 1]==192) * (img[:,:,2]==0)] = 18
    img[(img[:,:,0]==0) * (img[:, :, 1]==192) * (img[:,:,2]==128)] = 19
    img[(img[:,:,0]==128) * (img[:, :, 1]==64) * (img[:,:,2]==0)] = 20

    img[img[:, :, 0] > 20] = 0 # Edge as Background
    return img


def convert_segmmentations_VOC():
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "converted")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "converted"))
    for image_dir in os.listdir(os.path.join(VOC_DATA_PATH, "SegmentationClass")):
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "SegmentationClass", image_dir), 1)
        img = convert_VOC(img)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "converted", image_dir),img)


def make_dataset_VOC():
    '''
    Make necessary dirs
    '''
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "test")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "test"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train", "imgs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train", "imgs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "train", "segs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "train", "segs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "test", "imgs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "test", "imgs"))
    if not os.path.exists(os.path.join(VOC_DATA_PATH, "test", "segs")):
        os.makedirs(os.path.join(VOC_DATA_PATH, "test", "segs"))
    ''' End '''
    train_list = open(VOC_TRAIN_TXT, "r").readlines()
    test_list = open(VOC_VAL_TXT, "r").readlines() 
    for image_name in train_list:
        image_name = image_name.strip()
        # Save images
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "JPEGImages", image_name+".jpg"),1)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "train", "imgs", image_name+".jpg"), img)
        # Save segmentations
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "converted", image_name+".png"))
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "train", "segs", image_name+".png"), img)
    for image_name in test_list:
        image_name = image_name.strip()
        # Save images
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "JPEGImages", image_name+".jpg"),1)
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "test", "imgs", image_name+".jpg"), img)
        # Save segmentations
        img = cv2.imread(os.path.join(VOC_DATA_PATH, "converted", image_name+".png"))
        cv2.imwrite(os.path.join(VOC_DATA_PATH, "test", "segs", image_name+".png"), img)
    
    

if __name__ == "__main__":
    convert_segmentations_CUB()