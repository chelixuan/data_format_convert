import os
import json
from tqdm import tqdm
from collections import Counter

CHN2ENG = {
    '行人': 'person', 
    '公交车': 'bus', 
    '小汽车': 'car', 
    '自行车': 'bicycle', 
}
ENG2CHN = dict(zip(CHN2ENG.values(),CHN2ENG.keys()))


def get_specific_categories_json(raw, categories, save, folder=None):
    f = open(raw_json)
    info = json.load(f)

    raw_images = info['images']
    raw_cate = info['categories']
    raw_ann = info['annotations']

    # get new categories 外包标注，中文类别名称
    cateid_convert = {}
    new_cate = []
    for t_cate in categories:
        t_chn_cate = ENG2CHN[t_cate]
        t_cate_id = [x['id'] for x in raw_cate if x['name'] == t_chn_cate][0]
        t_cate_id_new = int(categories.index(t_cate) + 1)

        cateid_convert[t_cate_id] = t_cate_id_new
        new_cate.append(
            {
                'id': t_cate_id_new,
                'name': t_cate,
            }
        )


    # # 自己标注，英文类别名称
    # cateid_convert = {}
    # new_cate = []
    # for t_cate in categories:
    #     t_cate_id = [x['id'] for x in raw_cate if x['name'] == t_cate][0]
    #     t_cate_id_new = int(categories.index(t_cate) + 1)

    #     cateid_convert[t_cate_id] = t_cate_id_new
    #     new_cate.append(
    #         {
    #             'id': t_cate_id_new,
    #             'name': t_cate,
    #         }
    #     )

    # get new annotations
    new_ann = []
    image_ids = []
    print('\nget new annotations : \n')
    for t_ann in tqdm(raw_ann):
        raw_cate_id = t_ann['category_id']
        if raw_cate_id in list(cateid_convert.keys()):
            image_id = t_ann['image_id']
            image_ids.append(image_id)

            t_ann_new = {
                'id': t_ann['id'],
                'image_id' : image_id,
                'category_id': cateid_convert[raw_cate_id],
                'area': t_ann['area'],
                'bbox': t_ann['bbox'],
                'iscrowd': 0,
            }
            new_ann.append(t_ann_new)

    # get new images
    new_images = []
    print('\nget new images : \n')
    for t_image in tqdm(raw_images):
        t_image_id = t_image['id']
        t_file_name = t_image['file_name']

        if t_image_id in image_ids:
            t_image_name = t_file_name[t_file_name.rfind('/')+1:]
            if folder == None:
                t_file_name_new = t_file_name[t_file_name.rfind('/', 0, t_file_name.rfind('/')) + 1:t_file_name.rfind('/')+1] + t_image_name
            else:
                t_file_name_new = folder + t_image_name
        
            t_image['file_name'] = t_file_name_new
            new_images.append(t_image)

    new_coco = dict()
    new_coco['images'] = new_images
    new_coco['annotations'] = new_ann
    new_coco['categories'] = new_cate

    with open(save, 'w') as f:
        json.dump(new_coco, f, indent=4)
        print('specific categories json has been writen to : ', save)

    print()
    print('num images : ', len(new_images))
    print('num bbox : ', len(new_ann))

    count_ann = Counter()
    for t_ann in tqdm(new_ann):
        t_cate_id = t_ann['category_id']
        t_cate_name = [x['name'] for x in new_cate if x['id'] == t_cate_id][0]

        count_ann[t_cate_name] += 1
    
    print()
    for t in categories:
        print('{} : {} '.format(t, count_ann[t]))
    print()
        

raw_json = '/mnt/e/dataset/images/annotations/raw_per_batch_annotations/train_trash_1021-1024_batch_4.json'
save_json = '/mnt/e/dataset/images/annotations/specific_categories/train_5trash_1008_C200_batch_3-2.json'
folder = 'train/train_batch_3/'

categories = ['person', 'bicycle',]

save_path = save_json[:save_json.rfind('/')]
os.makedirs(save_path, exist_ok=True)

get_specific_categories_json(raw_json, categories, save_json, folder)



