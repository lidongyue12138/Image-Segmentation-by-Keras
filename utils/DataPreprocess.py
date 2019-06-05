import cv2
import os

images_path = "./Datasets/CUB_200_2011/images/"
segs_path = "./Datasets/CUB_200_2011/segmentations/"
save_path = "./Datasets/CUB_200_2011/converted/"

CUB_DATA_PATH = "./Datasets/CUB_200_2011/"
IMAGE_LIST_DIR = "./Datasets/CUB_200_2011/images.txt"
TRAIN_TEST_DIR = "./Datasets/CUB_200_2011/train_test_split.txt"

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

        image_name = os.path.basename(image_dir)
        if ifTrain:
            # Save images
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "images", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "train", "imgs", image_name), img)
            # Save segmentations
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "converted", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "train", "segs", image_name), img)
        else:
            # Save images
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "images", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "test", "imgs", image_name), img)
            # Save segmentations
            img = cv2.imread(os.path.join(CUB_DATA_PATH, "converted", image_dir))
            cv2.imwrite(os.path.join(CUB_DATA_PATH, "test", "segs", image_name), img)
        

if __name__ == "__main__":
    make_dataset_CUB()