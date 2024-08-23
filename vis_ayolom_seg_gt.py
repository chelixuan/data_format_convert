import os
import cv2
import random
from tqdm import tqdm

image_path = '/mnt/e/dataset/multi_task_dataset/images/train/'
drivable_txt_save_path = '/mnt/e/dataset/multi_task_dataset/seg-drivable/labels/train/'
lane_txt_save_path = '/mnt/e/dataset/multi_task_dataset/seg-lane/labels/train/'
vis_save_path = '/mnt/e/dataset/multi_task_dataset/vis_train_seg/'

os.makedirs(vis_save_path, exist_ok=True)

images = os.listdir(image_path)
# images = random.sample(images, 50)

for image_name in tqdm(images):
    flag_dri, flag_lane = False, False

    txt_name = image_name[:-4] + '.txt'
    img = cv2.imread(image_path + image_name)
    image_height, image_width, _ = img.shape

    # if (not os.path.exists(drivable_txt_save_path + txt_name)) or (not os.path.exists(lane_txt_save_path + txt_name)):
    #     continue

    if os.path.exists(drivable_txt_save_path + txt_name):
        with open(drivable_txt_save_path + txt_name, 'r') as f:
            seg_drivable_info = f.read()
            seg_drivable_info = seg_drivable_info.split('\n')
            seg_drivable_info = [x for x in seg_drivable_info if len(x) > 0]

        # print('num_drivable : ', len(seg_drivable_info))
        for t_info in seg_drivable_info:
            t_info = t_info.split(' ')

            # color = [random.randint(0, 255) for _ in range(3)]
            color = (0, 0, 255)

            for index in range(1, len(t_info), 2):
                flag_dri = True

                x = float(t_info[index])
                y = float(t_info[index + 1])

                point_x, point_y = image_width * x, image_height * y
                point_x, point_y = int(point_x), int(point_y)

                point_x, point_y = min(point_x, image_width), min(point_y, image_height)
                point_x, point_y = max(0, point_x), max(0, point_y)
                

                if index == 1:
                    init_x, init_y = point_x, point_y
                    start_x, start_y = point_x, point_y
                    end_x, end_y = point_x, point_y
                
                else:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = point_x, point_y
                
                cv2.circle(img, (point_x, point_y), 5, color, -1)
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)

                if index == len(t_info) - 2:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = init_x, init_y

                    cv2.circle(img, (point_x, point_y), 5, color, -1)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)
    
    if os.path.exists(lane_txt_save_path + txt_name):
        with open(lane_txt_save_path + txt_name, 'r') as f:
            seg_lane_info = f.read()
            seg_lane_info = seg_lane_info.split('\n')
            seg_lane_info = [x for x in seg_lane_info if len(x) > 0]

        # print('num_lane', len(seg_lane_info))
        for t_info in seg_lane_info:
            t_info = t_info.split(' ')

            # color = [random.randint(0, 255) for _ in range(3)]
            color = (0, 255, 0)
            for index in range(1, len(t_info), 2):
                flag_lane = True

                x = float(t_info[index])
                y = float(t_info[index + 1])

                point_x, point_y = image_width * x, image_height * y
                point_x, point_y = int(point_x), int(point_y)

                if index == 1:
                    init_x, init_y = point_x, point_y
                    start_x, start_y = point_x, point_y
                    end_x, end_y = point_x, point_y
                
                else:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = point_x, point_y
                # else:
                #     start_x, start_y = end_x, end_y
                #     end_x, end_y = init_x, init_y
                
                cv2.circle(img, (point_x, point_y), 5, color, -1)
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)

                if index == len(t_info) - 2:
                    start_x, start_y = end_x, end_y
                    end_x, end_y = init_x, init_y

                    cv2.circle(img, (point_x, point_y), 5, color, -1)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)

    # cv2.imwrite(vis_save_path + image_name, img)

    if flag_dri or flag_lane:
        cv2.imwrite(vis_save_path + image_name, img)
    else:
        print('none seg : ', image_name)

