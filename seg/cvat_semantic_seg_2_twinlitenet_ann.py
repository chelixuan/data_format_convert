import os
import cv2
import numpy as np
from tqdm import tqdm

raw_path = '/home/chelx/dataset/seg_images/1028_add_bad_case/SegmentationClass/basicfinder/pool_seg/lyon2024_seg_1028add_bad_case/'
save_root = '/home/chelx/dataset/seg_images/1028_add_bad_case/'
save_floor = save_root + 'floor/'
save_wall = save_root + 'wall/'

os.makedirs(save_floor, exist_ok = True)
os.makedirs(save_wall, exist_ok = True)

# temp  --------------------------------------------
raw_images = os.listdir("/home/chelx/dataset/seg_images/images/lyon2024_seg_1028add_bad_case/")
exists_images = [x[:-4] + ".jpg" for x in os.listdir("/home/wybj/chelx/dataset/seg_images/1028_add_bad_case/floor/")]

images = [x for x in raw_images if x not in exists_images]

print(len(raw_images))
print(len(exists_images))

print(raw_images[0])
print(exists_images[0])

print(len(images))
print("images : ", images)
# --------------------------------------------------

for t_image in tqdm(images):
    img = cv2.imread(raw_path + t_image)
    # background: (0, 0, 0), wall: (67, 91, 185), floor: (33, 244, 51)
    floor = np.zeros(img.shape)
    wall = np.zeros(img.shape)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            pixel = img[h][w]
            # wall
            if tuple(pixel) == (67, 91, 185):
                wall[h][w] = [255, 255, 255]
            # floor
            elif tuple(pixel) == (33, 244, 51):
                floor[h][w] = [255, 255, 255]
    cv2.imwrite(save_floor + t_image, floor)
    cv2.imwrite(save_wall + t_image, wall)

            