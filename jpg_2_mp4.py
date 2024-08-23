import cv2
import os
import numpy as np
from tqdm import tqdm
        
def jpg_2_mp4(image_path, images, save_mp4_path, fps=15, frameSize=(1920,1088)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowrite = cv2.VideoWriter(save_mp4_path, fourcc, fps, frameSize)

        for image in tqdm(images):
                img = cv2.imread(image_path + image)
                assert img is not None, "image error ~~~ "
                videowrite.write(img)
        print('mp4 has been successfully saved in : {} \n'.format(save_mp4_path))

      
def main():
        # image_path = '/mnt/d/charging_onnx_models/use_camera/h265_videos/h265_videos/images/'
        image_path = '/home/chelx/M2_model/test_video_images/'

        mp4_save_root = '/home/chelx/M2_model/'
        os.makedirs(mp4_save_root, exist_ok=True)
        # mp4_name = image_path[image_path[:-1].rfind('/')+1:-1] + '.mp4'
        mp4_name = 'outlet_test_video.mp4'
        mp4_save_path = mp4_save_root + mp4_name

        # images = ['2023-06-01_11-04-37.h265_' + str(i) + '.jpg' for i in range(9453, 9574)]
        images = [str(i) + '.jpg' for i in range(1036, 13293)]

        jpg_2_mp4(image_path, images, mp4_save_path, frameSize=(1920, 1080))


if __name__ == "__main__":
        main()


