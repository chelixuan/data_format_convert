# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from tqdm import tqdm

def save_img(video_path, videos, save_path, timeF = 10):
    # count = 0

    # for video_name in videos:
    for i in tqdm(range(len(videos))):
        count = 0

        video_name = videos[i]
        print(i+1, '/', len(videos), ' --> ')

        vc = cv2.VideoCapture(video_path+video_name)
        print('video : ', video_path+video_name)
        
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
            print('Attention : There is something wrong ... \n')
 
        # timeF = 10
        c = 0

        while rval:
            rval, frame = vc.read()
            #pic_path = folder_name + '/'
            if (c % timeF == 0):
                # video_path
                # t_name = video_name[:video_name.rfind('.')].replace(' ', '_')

                # video_root
                t_name = video_name[video_name.find('/')+1:video_name.rfind('.')].replace(' ', '_')
                
                # save_name = t_name + '-' + str(count) + '.jpg'
                save_name = str(count) + '.jpg'
                # print(count, ' --> ', save_name)
                
                try:
                    # 逆时针旋转 90度 --> C200水下相机拍摄视频
                    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # 顺时针旋转90度
                    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    cv2.imwrite(save_path + save_name, frame)
                    count = count + 1
                except Exception as e:
                    print(e)
                    pass
                         
            c = c + 1
            cv2.waitKey(1)
        vc.release()
        print('get {} images ...\n'.format(count))

video_path = '/home/chelx/M2_model/'
# videos = os.listdir(video_path)
videos = ['test_video.mp4']

save_path = '/home/chelx/M2_model/test_video_images/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_img(video_path, videos, save_path, timeF=1)
print()
print('Finished ~~')

