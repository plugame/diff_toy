import os
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np


root_dir = r"C:\lab\mori_m\siage\image"
line_dir = r"C:\lab\mori_m\siage\extract_line"
out_dir = r"C:\lab\mori_m\siage\separate_cut2"

image_path_list = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]

image_path_set = set(image_path_list)

cut_name_list = []

#i=Image.open(r"C:\lab\mori_m\siage\separate_cut2\C002_B_image\1_C002_B002.png")

def is_all_white(pil_image):
    # RGB などに変換しておく（白の判定がしやすいように）
    image = pil_image.convert("RGB")
    np_img = np.array(image)

    # 白色RGB値は (255, 255, 255)
    return np.all(np_img == 255)


def reft_wright_crop_image(image_path,dst_dir):
    basename = os.path.basename(image_path)
    image = Image.open(image_path)
    w, h = image.size
    box = (0,0,h,h)
    for i,(a,b) in enumerate([(0,h),(w-h,w)]):
        box = (a,0,b,h)
        cropped_img = image.crop(box)
        cropped_img = cropped_img.resize((512,512))
        if not is_all_white(cropped_img):
            cropped_img.save(os.path.join(dst_dir,f"{i}_{basename}"))

data_num = 0
for path in tqdm(image_path_list):
    cut_name = path[:-7]
    if cut_name in set(cut_name_list):
        continue

    cut_name_list.append(cut_name)
    count = sum(cut_name in s for s in image_path_list)
    if count >= 3:
        out_sub_dir = os.path.join(out_dir,os.path.basename(cut_name))
        os.makedirs(out_sub_dir+"_image",exist_ok=True)
        os.makedirs(out_sub_dir+"_line",exist_ok=True)
        for image_path in image_path_list:
            if image_path[:-7] == cut_name:
                line_path = os.path.join(line_dir,os.path.basename(image_path))
                data_num += 1
                #shutil.copy(p,os.path.join(out_sub_dir,os.path.basename(p)))
                reft_wright_crop_image(image_path,out_sub_dir+"_image")
                reft_wright_crop_image(line_path,out_sub_dir+"_line")

print(data_num*2)
