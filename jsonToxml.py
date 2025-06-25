# import json
# import os
#
#
# def convert_to_yolo(annotations_dir, images_dir, output_dir):
#     for annotation_file in os.listdir(annotations_dir):
#         if annotation_file.endswith('.json'):  # 根据实际情况修改后缀名
#             with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
#                 data = json.load(f)
#
#             for image_data in data['images']:
#                 image_name = image_data['file_name']
#                 image_path = os.path.join(images_dir, image_name)
#
#                 image_width = image_data['width']
#                 image_height = image_data['height']
#
#                 yolo_annotations = []
#
#                 # Find annotations for current image
#                 for annotation in data['annotations']:
#                     if annotation['image_id'] == image_data['id']:
#                         class_id = annotation['category_id']
#                         bbox = annotation['bbox']
#                         # x_center = (bbox[0] + bbox[2]) / 2.0 / image_width
#                         # y_center = (bbox[1] + bbox[3]) / 2.0 / image_height
#                         x_center = (bbox[0]) / 2.0 / image_width
#                         y_center = (bbox[1]) / 2.0 / image_height
#                         width = (bbox[2]) / image_width
#                         height = (bbox[3]) / image_height
#
#                         yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
#
#                 output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
#                 with open(output_file, 'w') as out_f:
#                     out_f.write("\n".join(yolo_annotations))
#
#
# # Example usage:
# annotations_dir = r'D:\dl\RUOD\RUOD_pic\1'
# images_dir = r'D:\dl\RUOD\RUOD_pic\test'
# output_dir = r'D:\dl\RUOD\RUOD_pic\label_test'
#
# convert_to_yolo(annotations_dir, images_dir, output_dir)

# from PIL import Image
#
# # 设置JSON文件路径、图像文件夹路径、输出目录路径和类名列表
# json_file = r'D:\dl\RUOD\RUOD_ANN\instances_train.json'
# images_dir = r'D:\dl\RUOD\RUOD_pic\train'
# output_dir = r'D:\dl\RUOD\RUOD_pic\labels'
# classes = ['holothurian', 'echinus', 'scallop', 'tarfish', 'fish', 'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish']  # 替换为实际的类名
#
# # 执行转换
# convert_json_to_yolo(json_file, images_dir, output_dir, classes)

import json

def coco_to_yolo(coco_json_file, output_dir, class_map=None):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    categories = coco_data['categories']
    images = coco_data['images']
    annotations = coco_data['annotations']

    if class_map is None:
        class_map = {category['id']: category['name'] for category in categories}

    for image in images:
        image_id = image['id']
        file_name = image['file_name']
        image_width = image['width']
        image_height = image['height']

        yolo_annotations = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                class_name = class_map[category_id]
                bbox = annotation['bbox']
                x_min, y_min, width, height = bbox
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                width /= image_width
                height /= image_height
                yolo_annotations.append(f"{class_name} {x_center} {y_center} {width} {height}")

        if len(yolo_annotations) > 0:
            output_file_path = output_dir + file_name.replace('.jpg', '.txt')
            with open(output_file_path, 'w') as output_file:
                for annotation in yolo_annotations:
                    output_file.write(annotation + '\n')

# Example usage
coco_json_file = 'D:\dl\RUOD\RUOD_ANN\instances_test.json'
output_dir = 'D:/dl/RUOD/1/label_test/'
class_map = {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}  # Define your class map here
coco_to_yolo(coco_json_file, output_dir, class_map)
