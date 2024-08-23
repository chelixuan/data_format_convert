import os
import cv2
import numpy as np
from tqdm import tqdm

raw_path = '/home/chelx/dataset/multi_task_dataset/bdd_format/images/train/'
seg_path = '/home/chelx/dataset/multi_task_dataset/train/SegmentationClass/'
save_root = '/home/chelx/dataset/multi_task_dataset/bdd_format/'

floor_path = save_root + 'drivable_segmentation/train/'
facade_path = save_root + 'lane_segmentation/train/'

os.makedirs(floor_path, exist_ok=True)
os.makedirs(facade_path, exist_ok=True)

files = os.listdir(raw_path)
files = [x for x in files if x[-4:] == '.jpg']

for t_file in tqdm(files):
    t_file = files[i]
    
    raw = cv2.imread(raw_path + t_file)

    floor = np.zeros(raw.shape)
    facade = np.zeros(raw.shape)

    seg = seg_path + t_file[:-4] + '.png'

    if os.path.exists(seg):
        img = cv2.imread(seg)

        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                if list(img[h][w]) == [67, 56, 72]:
                    # floor[h][w] = np.array([67, 56, 72])
                    floor[h][w] = np.array([127, 127, 127])
                elif list(img[h][w]) == [11, 129, 77]:
                    # facade[h][w] = np.array([11, 129, 77])
                    facade[h][w] = np.array([255, 255, 255])
    
    cv2.imwrite(floor_path + t_file, floor)
    cv2.imwrite(facade_path + t_file, facade)

    

