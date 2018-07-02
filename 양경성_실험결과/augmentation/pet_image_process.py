import cv2
import copy
import os

def make_simple_image(ori_img,weight,black_limit,step):
    img_c = copy.deepcopy(ori_img)
    x, y, c = ori_img.shape
    for i in range(x):
        for j in range(y):
            for n in range(0,len(step)):
                if ori_img[i,j,0] <= black_limit :
                    img_c[i,j]=0
                    break
                elif ori_img[i,j,0] < int(step[n]) :
                    img_c[i,j]=min(255,step[n]+weight)
                    break
    return img_c

weight = 20
black_limit = 50
step=[black_limit+10,black_limit+30,black_limit+50,black_limit+70,black_limit+80,black_limit+90,black_limit+100,black_limit+110,black_limit+120,black_limit+130,255]

# fname1 = '1gr_45_0001.jpg'
# fname2 = '2gr_45_0001.jpg'
# fname3 = '3gr_45_0001.jpg'

source_dir = './origin/1gr'
save_dir = './simple/1gr'
if not os.path.isdir(save_dir): os.mkdir(save_dir)

file_list = os.listdir(source_dir)
print(file_list)
for i in range(len(file_list)):
    fname = source_dir+'/'+file_list[i]
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img_c = make_simple_image(img,weight,black_limit,step)
    # cv2.imshow('manuplate 1gr', img_c)
    new_name=save_dir+'/'+file_list[i]
    cv2.imwrite(new_name, img_c)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

# img1 = cv2.imread(fname1, cv2.IMREAD_COLOR)
# img1_c = make_simple_image(img1,weight,black_limit,step)
# img2 = cv2.imread(fname2, cv2.IMREAD_COLOR)
# img2_c = make_simple_image(img2,weight,black_limit,step)
# img3 = cv2.imread(fname3, cv2.IMREAD_COLOR)
# img3_c = make_simple_image(img3,weight,black_limit,step)
# cv2.imshow('Original 1gr', img1)
# cv2.imshow('manuplate 1gr', img1_c)
# cv2.imshow('Original 2gr', img2)
# cv2.imshow('manuplate 2gr', img2_c)
# cv2.imshow('Original 3gr', img3)
# cv2.imshow('manuplate 3gr', img3_c)