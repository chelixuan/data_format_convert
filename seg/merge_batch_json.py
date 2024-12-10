import os
import json
from tqdm import tqdm
import prettytable as pt

json_root = "/home/chelx/dataset/seg_images/coco_annotations/standard_coco_annos/"
merge_jsons = [x for x in os.listdir(json_root) if "train" in x]
save_json = "/home/chelx/dataset/seg_images/coco_annotations/train_20241118.json"

# merge_jsons = [x for x in os.listdir(json_root) if "val" in x]
# save_json = "/home/chelx/dataset/seg_images/coco_annotations/val_20241118.json"

save_label_contribution_txt = save_json[:-5] + "_labels.txt"

standard_cates = [
        {"id": 1, "name": "step", "supercategory": "step"},
        {"id": 2, "name": "facade", "supercategory": "step"},
        {"id": 3, "name": "floor", "supercategory": "floor"},
        {"id": 4, "name": "slope", "supercategory": "floor"},
        {"id": 5, "name": "wall", "supercategory": "wall"},
        {"id": 6, "name": "bevel", "supercategory": "wall"},
    ]

total_images, total_annotations = [], []
count_image, count_anno = 0, 0

count_per_cate = [0]*len(standard_cates)
tb = pt.PrettyTable()
tb.field_names = ["category_id", "category_name", "num"]

for t_json in merge_jsons:
    json_path = json_root + t_json
    info = json.load(open(json_path))

    images = info["images"]
    annos = info["annotations"]

    print(f"\n{t_json} : ")
    print(f"collect images : ")
    image_id_convert = dict()
    for t_image in tqdm(images):
        image_id = t_image["id"]
        image_id_new = count_image

        t_image["id"] = image_id_new
        total_images.append(t_image)
        count_image += 1

        image_id_convert[image_id] = image_id_new
    print(f"collect annos : ")
    for t_ann in tqdm(annos):
        raw_image_id = t_ann["image_id"]
        image_id_new = image_id_convert[raw_image_id]
        raw_ann_id = t_ann["id"]
        ann_id_new = count_anno

        t_ann["image_id"] = image_id_new
        t_ann["id"] = ann_id_new
        total_annotations.append(t_ann)
        count_anno += 1

        count_per_cate[int(t_ann["category_id"] - 1)] += 1

print(f"\nmerge finish : ")
print("num_images = ", len(total_images))
print("num_anno = ", len(total_annotations))

print(f"\nlabel contributions : ")
for t_cate in standard_cates:
    cate_id = t_cate["id"]
    cate_name = t_cate["name"]
    cate_num = count_per_cate[int(cate_id - 1)]
    tb.add_row([cate_id, cate_name, cate_num])

with open(save_label_contribution_txt, "w") as f:
    f.write(str(tb))
print(tb)

info_new = dict()
info_new["images"] = total_images
info_new["categories"] = standard_cates
info_new["annotations"] = total_annotations

with open(save_json, "w") as f:
    json.dump(info_new, f, indent=4)
    print("\nsuccess ~~ \n")

