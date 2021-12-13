from PIL import Image
import os
 
path = './test data/LR'
all_images = os.listdir(path)
# print(all_images)
 
for image in all_images:
    image_path = os.path.join(path, image)
    img = Image.open(image_path)  # 打开图片
    img = img.convert("RGB")  # 4通道转化为rgb三通道
    save_path = './test data/RGB'
    img.save(save_path + image)

