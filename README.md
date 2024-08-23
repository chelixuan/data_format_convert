# 常用数据格式准换代码

## 标准检测数据格式准换
 - coco2yolo
 - voc2coco
 - coco2bdd

## 标注工具格式转换
 - labelme 分割转 coco
 - labelimg 检测转 yolo

## groundtruth 可视化
 - 检测
    -+ coco 格式 gt 可视化
    -+ yolo 格式 gt 可视化 
 - 分割
    -+ 轮廓：类似于 yolo txt 格式（A-YOLOM 算法使用的格式） gt 可视化
    -+ 区域：将整图区域分割拆分成每个类别的区域分割 gt
    -+ todo 轮廓：coco 格式 gt 可视化
 - 关键点
    -+ yolo 格式 landmarks gt 可视化
