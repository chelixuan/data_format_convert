import os
import random
import shutil
from tqdm import tqdm

root_path = '/home/wybj/chelx/dataset/Lyon2024/dataset/slam_gt/'
save_root = '/home/wybj/chelx/dataset/Lyon2024/dataset/lyon2024_tag/'
os.makedirs(save_root, exist_ok=True)

train_root = save_root + 'train/'
os.makedirs(train_root, exist_ok=True)
val_root = save_root + 'val/'
os.makedirs(val_root, exist_ok=True)

# train_root = save_root + 'all_train/'
# os.makedirs(train_root, exist_ok=True)

def get_all_image(root_path):
    all_images = []

    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            extension = file_path.rpartition('.')[2].lower()
            if extension in image_extensions:
                all_images.append(file_path)

    return all_images

def slamtag_2_retina(label_path):
    f = open(label_path)
    info = f.read().strip().split('\n')
    tag_info = info[1]
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y = tag_info.split()
    p1_x, p1_y = float(p1_x), float(p1_y)
    p2_x, p2_y = float(p2_x), float(p2_y)
    p3_x, p3_y = float(p3_x), float(p3_y)
    p4_x, p4_y = float(p4_x), float(p4_y)

    box_x1, box_y1 = min(p1_x, p2_x, p3_x, p4_x), min(p1_y, p2_y, p3_y, p4_y)
    box_x2, box_y2 = max(p1_x, p2_x, p3_x, p4_x), max(p1_y, p2_y, p3_y, p4_y)
    w, h = box_x2 - box_x1, box_y2 - box_y1
    retina_info = [box_x1, box_y1, w, h, p1_x, p1_y, 0, p2_x, p2_y, 0, p3_x, p3_y, 0, p4_x, p4_y, 0]
    retina_line = ' '.join(map(str, retina_info))

    # print('tag_info = ', tag_info)
    # print()
    # print(retina_line)
    return retina_line + '\n'


all_images = get_all_image(root_path)
# all_images = [x.lstrip(root_path) for x in all_images]

# image = all_images[20]
# label = image.replace('image', 'point').replace('.jpg', '.txt')
# print()
# print(image)
# print(label)
# print(os.path.exists(image))
# exit()

# 10% val
val_images = random.sample(all_images, int(0.1*len(all_images)))
train_images = [x for x in all_images if x not in val_images]

val_image_save_root = val_root + 'images/'
val_txt_path = val_root + 'label.txt'
print('\nget val data : \n')
with open(val_txt_path, 'w') as f:
    for image in tqdm(val_images):
        # image_path = image.lstrip(root_path) # 相对数据根目录的地址
        # 上面写法会额外去掉 root_path 中含有的字符
        image_path = image.replace(root_path, "")

        image_sub_path = image_path[:image_path.rfind('/')+1] # 相对根目录的路径
        image_save_path = val_image_save_root + image_sub_path
        os.makedirs(image_save_path, exist_ok=True)

        image_name = image_path[image_path.rfind('/')+1:]

        shutil.copy(image, image_save_path + image_name)

        label = image_path.replace('image', 'point').replace('.jpg', '.txt')

        retina_line = slamtag_2_retina(root_path + label)

        image_line = '# ' + image_path + '\n'

        f.write(image_line)
        f.write(retina_line)


train_image_save_root = train_root + 'images/'
train_txt_path = train_root + 'label.txt'
print('\nget train data : \n')
with open(train_txt_path, 'w') as f:
    for image in tqdm(train_images):
        # image_path = image.lstrip(root_path) # 相对数据根目录的地址
        # 上面写法会额外去掉 root_path 中含有的字符
        image_path = image.replace(root_path, "")
        image_sub_path = image_path[:image_path.rfind('/')+1] # 相对根目录的路径
        image_save_path = train_image_save_root + image_sub_path
        os.makedirs(image_save_path, exist_ok=True)

        image_name = image_path[image_path.rfind('/')+1:]

        shutil.copy(image, image_save_path + image_name)

        label = image_path.replace('image', 'point').replace('.jpg', '.txt')
        retina_line = slamtag_2_retina(root_path + label)

        image_line = '# ' + image_path + '\n'

        f.write(image_line)
        f.write(retina_line)
