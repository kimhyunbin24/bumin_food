import os
import numpy as np

# root = r'F:\AI hub data\음식 이미지 및 영양정보 텍스트\food data\라벨 일부\mask'
root = r'F:\AI hub data\음식 이미지 및 영양정보 텍스트\re mask\mask'

for category in os.listdir(root):
    print(category)
    ctg = int(category.split('.')[0]) + 1
    print(ctg)
    category_path = os.path.join(root, category)

    for sub in os.listdir(category_path):
        sub_path = os.path.join(category_path, sub)

        for label_file in os.listdir(sub_path):
            if label_file.endswith('.npy'):
                label_file_path = os.path.join(sub_path, label_file)
                # print(label_file_path)

                arr = np.load(label_file_path)
                print('origianl: ', np.unique(arr))
                arr[arr == 2] = ctg
                print('changed: ', np.unique(arr))