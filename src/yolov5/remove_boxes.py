import os
import cv2
import glob

label_paths = glob.glob("runs/detect/v5x6_1280_tta_test/labels/*")
img_path = '/mnt/sda/shadow_user/nguyen.hung.quang/compe/docker/src/RELEASE_private_test/pill/image/'
# print(label_paths)
os.makedirs("runs/detect/v5x6_1280_tta_test_rm_box/labels", exist_ok=True)
for l_path in label_paths:
    # if "7e2a87d0194edc10855f" not in l_path:
    #     continue
    f = open(l_path, "r")
    
    img = cv2.imread(img_path + os.path.basename(l_path)[:-3] + 'jpg')
    if img is None:
        img = cv2.imread(img_path + os.path.basename(l_path)[:-3] + 'JPG')
    h, w, _ = img.shape
    data = f.readlines()
    for item in data:
        c, coff, x1, y1, x2, y2 = item.split(" ")
        c = int(c)
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        # center_X = float(center_X)
        # center_y = float(center_y)
        # width = float(width)
        # height = float(height)
        # coff = float(coff)
        # x1 = (center_X-width/2)
        # x2 = (center_X+width/2)
        # y1 = (center_y-height/2)
        # y2 = (center_y+height/2)
        # print(x1, y1, x2, y2)
        norm_x1 = x1 / w
        norm_x2 = x2 / w
        norm_y1 = y1 / h
        norm_y2 = y2 / h
        
        if (norm_x1<1e-4 or norm_y1<1e-4 or norm_x2>1-1e-4 or norm_y2>1-1e-4):
            print(l_path)
            continue
        
        with open(l_path.replace("v5x6_1280_tta_test", "v5x6_1280_tta_test_rm_box"), "a") as f1:
            f1.write("{} {} {} {} {} {}\n".format(c, coff, int(x1), int(y1), int(x2), int(y2)))
    f.close()