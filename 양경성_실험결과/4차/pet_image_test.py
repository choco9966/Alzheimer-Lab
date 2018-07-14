import cv2
import copy


level=10
weight = 2
black_limit = 50
# white_limit = 130
st = int(256/level)

fname1 = '1gr_45_0001.jpg'
fname2 = '2gr_45_0001.jpg'
fname3 = '3gr_45_0001.jpg'

img1 = cv2.imread(fname1, cv2.IMREAD_COLOR)
img1_c = copy.deepcopy(img1)
x,y,c = img1.shape

for i in range(x):
    for j in range(y):
        # print(img[i,j,0])
        for n in range(1,level):
            # print(i,j,'2**n:',2**n)
            if img1[i,j,0] <= black_limit : #break
                # img1_c.itemset((i, j, 0), 0)
                img1_c[i,j]=0
                break
            # elif img1[i,j,0] >= white_limit : #break
            #     print(img1[i,j,0])
            #     img1_c[i,j]=(0,255,0)
            #     break
            elif img1[i,j,0] < st*n :
                # img1_c.itemset((i,j,0),min(255,st*(n+weight)))
                img1_c[i,j]=min(255,st*(n+weight))
                break

img2 = cv2.imread(fname2, cv2.IMREAD_COLOR)
img2_c = copy.deepcopy(img2)
x,y,c = img2.shape

for i in range(x):
    for j in range(y):
        # print(img[i,j,0])
        for n in range(1,level):
            # print(i,j,'2**n:',2**n)
            if img2[i,j,0] <= black_limit : #break
                img2_c[i, j] = 0
                break
            # elif img2[i, j, 0] >= white_limit:  # break
            #     img2_c[i, j] = (0,255,0)
            #     print(img2[i, j, 0])
            #     break
            elif img2[i,j,0] < st*n :
                img2_c[i,j] = min(255,st*(n+weight))
                break

img3 = cv2.imread(fname3, cv2.IMREAD_COLOR)
img3_c = copy.deepcopy(img3)
x,y,c = img3.shape

for i in range(x):
    for j in range(y):
        # print(img[i,j,0])
        for n in range(1,level):
            # print(i,j,'2**n:',2**n)
            if img3[i,j,0] <= black_limit : #break
                img3_c[i, j] = 0
                break
            # elif img3[i,j,0] >= white_limit : #break
            #     img3_c[i,j]=(0,255,0)
            #     print(img3[i, j, 0])
            #     break
            elif img3[i,j,0] < st*n :
                img3_c[i,j] = min(255,st*(n+weight))
                break


cv2.imshow('Original 1gr', img1)
cv2.imshow('manuplate 1gr', img1_c)
cv2.imshow('Original 2gr', img2)
cv2.imshow('manuplate 2gr', img2_c)
cv2.imshow('Original 3gr', img3)
cv2.imshow('manuplate 3gr', img3_c)
cv2.waitKey(0)
cv2.destroyAllWindows()