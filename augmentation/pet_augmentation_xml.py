#========================================================
# augmentation
# pet 이미지 파일을 싸이즈 , 잘라내기, 회전, 잡음, 밝기 로 파일 갯수 늘리기
# 세그멘테이션(개체와 위치정보)된 xml 파일 지원(변형된 이미지에 맞게 자동 계산하여 xml 생성)
#========================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
import sys
from math import floor, ceil, pi
import random
import copy
import xml.etree.ElementTree as ET
import shutil

def read_xml(fname):
    # parse xml file
    doc = ET.parse(data_dir+'/'+fname)
    # get root node
    root = doc.getroot()
    xsets =[]
    for object in root.iter("object"):
        # print("name : ", object.findtext("name"))
        xname = object.findtext("name")
        for bndbox in object.iter("bndbox"):
            # print("xmin : ", bndbox.findtext("xmin"))
            # print("ymin : ", bndbox.findtext("ymin"))
            # print("xmax : ", bndbox.findtext("xmax"))
            # print("ymax : ", bndbox.findtext("ymax"))
            xpos=[bndbox.findtext("xmin"),bndbox.findtext("ymin"),bndbox.findtext("xmax"),bndbox.findtext("ymax")]
        xset = [xname, xpos]
        xsets.append(xset)
    # print(xsets)
    # sys.exit()
    return xsets

def get_image_paths(folder):
    files = []
    ori_files = os.listdir(folder)
    ori_files.sort()
    # print('ori_files: ', ori_files)
    # print('target_files: ', target_files)
    if len(target_files)>0:
        ori_files = []
        for i in range(len(target_files)):
            x = os.path.splitext(os.path.basename(target_files[i]))[0]+'.xml'
            ori_files.append(target_files[i])
            ori_files.append(x)
            # print(target_files[i])
        ori_files.sort()
        # print('ori_files: ',ori_files)

    xml_sets =[]

    for i in range(len(ori_files)):
        d = os.path.splitext(os.path.basename(ori_files[i]))[1]
        if d=='.jpg' or d=='.JPG':
            files.append(ori_files[i])
        elif d=='.xml' or d=='.XML':
            if using_xml == 1:
                xsets = read_xml(ori_files[i])
                xml_sets.append(xsets)
    # print(xml_sets)

    files = ['{}/{}'.format(folder, file) for file in files]
    return files, xml_sets

def imcrop(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                       (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0, 0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

def translate_images(noise=[0,90],light=[0,100],rate=4):
    t_scales=[]
    cnt = 0
    for index in range(Max_scale*100, Min_scale*100, -50):
        t_scales.append(index * 0.0001)
    if debug >= 2: print(t_scales, len(t_scales))
    scales =[]
    X_scale_data = []
    iter = int(40 - (rate*5))
    if rate<=4:
        count=int(len(t_scales)/4 *rate)
        tt_scales = random.sample(t_scales, count)
        for i in range(count):
          scales.append(tt_scales[i])
        scales.sort()
        scales.reverse()
    else:
        return 0

    # for i in range(len(t_scales)):
    #     if rate==1 :
    #         if not(i % 4 == 1 or i % 4 == 2 or i % 4 == 3): scales.append(t_scales[i])
    #         iter = 40
    #     elif rate == 2:
    #         if not(i % 4 == 2 or i % 4 == 3) : scales.append(t_scales[i])
    #         iter = 30
    #     elif rate == 3:
    #         if not(i % 4 == 3) : scales.append(t_scales[i])
    #         iter = 25
    #     elif rate == 4:
    #         scales.append(t_scales[i])
    #         iter = 20
    #     else:
    #         return 0
    if debug >= 2: print(scales, len(scales))


    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        # img_data = X_imgs[k]
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        # img_data = copy.deepcopy(X_imgs[k])
        k_x1, k_y1 = 0, 0
        k_x2, k_y2 = img_data.shape[1], img_data.shape[0]
        # print(k_x1,k_y1,k_x2,k_y2)
        # print(xml_sets[k])
        # print(len(xml_sets[k]))
        if using_xml == 1:
            for i in range(len(xml_sets[k])):
                # print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # print('Ori xml', p_x1, p_y1, p_x2, p_y2)
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)
        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)

        for j in range(len(scales)):
            # scaled_img = cv2.resize(img_data, (0, 0), fx=scales[j], fy=scales[j])
            dx = k_x2 - int(k_x2 * scales[j])
            dy = k_y2 - int(k_y2 * scales[j])
            if dx==0: dx=2
            if dy==0: dy=2
            # for a in range(0,dx,int(dx/2)) :
            #     for b in range(0, dy, int(dy/2)):
            #         print(j, a, b, cnt)
            #         n_x1 = int(k_x1 * scales[j])+a
            #         n_y1 = int(k_y1 * scales[j])+b
            #         n_x2 = int(k_x2 * scales[j])+a
            #         n_y2 = int(k_y2 * scales[j])+b
            for a in range(0,2, 1) :
                for b in range(0, 2, 1):
                    n_x1 = int(k_x1 * scales[j]+a*(dx/4))
                    n_y1 = int(k_y1 * scales[j]+b*(dy/4))
                    n_x2 = int(k_x2 * scales[j]+a*(dx/4))
                    n_y2 = int(k_y2 * scales[j]+b*(dy/4))
                    # scaled_img = []
                    # print(n_x1,n_y1,n_x2,n_y2)
                    temp_img = imcrop(img_data, n_x1,n_y1,n_x2,n_y2)
                    scaled_img = copy.deepcopy(temp_img)
                    row, col, ch = scaled_img.shape
                    s_x1, s_y1 = 0, 0
                    s_x2, s_y2 = scaled_img.shape[1], scaled_img.shape[0]
                    if using_xml == 1:
                        if debug >= 2:
                            print('scaled img:', s_x1, s_y1, s_x2, s_y2)
                            print(k, len(xml_sets[k]))
                        new_pos=[]
                        for m in range(len(xml_sets[k])):
                            # print(xml_sets[k][i][1])
                            p_x1 = int(xml_sets[k][m][1][0])
                            p_y1 = int(xml_sets[k][m][1][1])
                            p_x2 = int(xml_sets[k][m][1][2])
                            p_y2 = int(xml_sets[k][m][1][3])
                            n_x1 = max(0,int(p_x1-a))
                            n_y1 = max(0,int(p_y1-b))
                            n_x2 = int(p_x2-a)
                            n_y2 = int(p_y2-b)
                            if debug >= 2: print(a,b,m,'Ori xml:',p_x1,p_y1,p_x2,p_y2, 'New xml:', n_x1,n_y1,n_x2,n_y2)
                            if n_x2 > s_x2:  n_x2 = s_x2
                            if n_y2 > s_y2:  n_y2 = s_y2

                            if 'numberplate'in xml_sets[k][m][0] : #m ==0:
                                if debug >= 2: print(xml_sets[k][m][0])
                                # print('***:',n_x2,s_x2,n_y2,s_y2)
                                if n_x2==s_x2 or n_y2==s_y2 :
                                    if debug >= 2: print('======Can t draw : because too small')
                                    new_pos.append([0, 0, 0, 0])
                                else:
                                    if not (n_x1 > s_x2 or n_y1 > s_y2):
                                        if debug >= 2: print(a, b, m, 'Ori:', p_x1, p_y1, p_x2, p_y2, 'Final_xml:', n_x1,
                                                             n_y1, n_x2, n_y2)
                                        # cv2.rectangle(scaled_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                                        new_pos.append([n_x1, n_y1, n_x2, n_y2])

                            else:
                                if not(n_x1 > s_x2 or n_y1 > s_y2) :
                                    if debug >= 2: print(a,b,m,'Ori:', p_x1, p_y1, p_x2, p_y2, 'Final_xml:', n_x1, n_y1, n_x2, n_y2)
                                    # cv2.rectangle(scaled_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                                    new_pos.append([n_x1, n_y1, n_x2, n_y2])
                                else:
                                    if debug >= 2: print('Can t draw')
                                    new_pos.append([0,0,0,0])

                    if not (noise[0] == 0 and noise[1] == 0):  # Add noise
                        for ratio in range(noise[0], noise[1], iter):
                            mean = 0
                            var = 0.05  # 0.00001  # 0.5
                            sigma = var ** 0.5
                            if grey_img : ch=1
                            gauss = np.random.normal(mean, sigma, (row, col, ch))
                            gauss = gauss.reshape(row, col, ch)
                            noisy_img = scaled_img + gauss * ratio
                            new_name = out_dir + '/' + ori_file + '_T%02d%02d_%02d_N%02d.jpg' % (a,b,j, ratio)
                            cv2.imwrite(new_name, noisy_img)
                            if using_xml == 1:
                                xml_name = ori_file + '_T%02d%02d_%02d_N%02d' % (a,b,j, ratio)
                                # if debug >= 2: print(new_pos)
                                Save_XML(k, xml_name, new_pos)
                            cnt += 1
                    if not (light[0] == 0 and light[1] == 0):  # Add Dark
                        for ratio in range(light[0], light[1], iter):
                            invGamma = 1.0 + ratio * 0.01
                            table = np.array([((i / 255.0) ** invGamma) * 255
                                              for i in np.arange(0, 256)]).astype("uint8")
                            gaussian_img = cv2.LUT(scaled_img, table)

                            new_name = out_dir + '/' + ori_file + '_T%02d%02d_%02d_L%02d.jpg' % (a,b,j, ratio)
                            cv2.imwrite(new_name, gaussian_img)
                            if using_xml == 1:
                                xml_name = ori_file + '_T%02d%02d_%02d_L%02d' % (a,b,j, ratio)
                                # if debug >= 2: print(new_pos)
                                Save_XML(k, xml_name, new_pos)
                            cnt += 1
                    if (noise[0] == 0 and noise[1] == 0 and light[0] == 0 and light[1] == 0):
                        new_name = out_dir + '/' + ori_file+'_T%02d%02d_%02d.jpg' % (a,b,j)
                        cv2.imwrite(new_name, scaled_img)
                        if using_xml == 1:
                            xml_name = ori_file+'_T%02d%02d_%02d' % (a,b,j)
                            # if debug >= 2: print(new_pos)
                            Save_XML(k,xml_name,new_pos)
                        cnt+=1
    return cnt

def Save_XML(f_no,xml_name, ksets):
    # print(ksets)
    ori_xml = os.path.splitext(os.path.basename(X_img_paths[f_no]))[0]
    # print(data_dir+'/'+ori_xml+'.xml')
    new_xml = out_dir+'/'+xml_name+'.xml'
    shutil.copy2(data_dir+'/'+ori_xml+'.xml', new_xml)

    tree = ET.parse(new_xml)
    root = tree.getroot()
    i=0
    for anno in root.iter("annotation"):
        for object in root.iter("object"):
            # print("name : ", object.findtext("name"))
            xname = object.findtext("name")
            for bndbox in object.iter("bndbox"):
                # print("xmin : ", bndbox.findtext("xmin"))
                # print("ymin : ", bndbox.findtext("ymin"))
                # print("xmax : ", bndbox.findtext("xmax"))
                # print("ymax : ", bndbox.findtext("ymax"))
                # xpos=[bndbox.findtext("xmin"),bndbox.findtext("ymin"),bndbox.findtext("xmax"),bndbox.findtext("ymax")]
                for xmin in object.iter("xmin"): xmin.text =str(ksets[i][0])
                for ymin in object.iter("ymin"): ymin.text =str(ksets[i][1])
                for xmax in object.iter("xmax"): xmax.text =str(ksets[i][2])
                for ymax in object.iter("ymax"): ymax.text =str(ksets[i][3])
            # if (ksets[i][2]==0 and ksets[i][3]==0): anno.remove(object)
            i+=1

    i=0
    for anno in root.iter("annotation"):
        for object in root.iter("object"):
            if (ksets[i][2] == 0 and ksets[i][3] == 0): anno.remove(object)
            i += 1
    tree.write(new_xml)
    # sys.exit()


def central_scale_images(noise=[0,90], light=[0,100], rate=4): #4 6 8 10
    t_scales=[]
    for index in range(Min_scale, Max_scale, 3):
        t_scales.append(index * 0.01)
    if debug >= 2: print(scale_list, len(scale_list))
    cnt=0
    # Various settings needed for Tensorflow operation
    scales =[]
    X_scale_data = []
    iter = int(40 - (rate*5))
    if rate<=4:
        count=int(len(t_scales)/4 *rate)
        tt_scales = random.sample(t_scales, count)
        for i in range(count):
          scales.append(tt_scales[i])
        scales.sort()
    else:
        return 0

    if debug >= 2: print(scales, len(scales))
    # sys.exit()
    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        # img_data = cv2.resize(img_data, (basewidth, hsize))
        # img_data =  #X_imgs[k]
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        # w_y, w_x,_ = X_imgs[k].shape
        # print(w_x, w_y)
        if using_xml == 1:
            if debug >= 2:
                print(xml_sets[k])
                print(len(xml_sets[k]))
            for i in range(len(xml_sets[k])):
                if debug >= 2:
                    print(xml_sets[k][i][0])
                    print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)

        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)
        for j in range(len(scales)):
            w_y, w_x, w_c = img_data.shape
            scaled_img = cv2.resize(img_data, (0, 0), fx=scales[j], fy=scales[j])
            row, col, ch = scaled_img.shape
            # 배율이 100%보다 적으면, 축소하고 검은색 이미지와 합치기
            if scales[j] <1.:
                temp_data = copy.deepcopy(img_data)
                temp_data[:,:,:]=0   #기존 크기의 검은색 이미지 생성
                x_offset = abs(int((w_x - col)/2.))
                y_offset = abs(int((w_y - row)/2.))
                temp_data[y_offset:y_offset + scaled_img.shape[0],
                x_offset:x_offset + scaled_img.shape[1]] = scaled_img
                # print(x_offset, y_offset, temp_data.shape)
                scaled_img = temp_data
                # scaled_img = copy.deepcopy(temp_data)
            # 배율이 100%보다 크면, 확대하고 동일한 크기로 중앙에서 자르기
            elif scales[j] >1.:
                x_offset = abs(int((w_x - col)/2.))
                y_offset = abs(int((w_y - row)/2.))
                temp_data = imcrop(scaled_img, x_offset, y_offset, x_offset+w_x, y_offset+w_y)
                # print(x_offset,y_offset, temp_data.shape)
                scaled_img = temp_data
                # scaled_img = copy.deepcopy(temp_data)
            row, col, ch = scaled_img.shape
            if using_xml == 1:
                #[보류] xml 좌표 계산 다시 해야 함 scales[j] 100%이하, 이상 구분 =>이하는 비율대로 축소, 이상은 이미지 사이즈대로,
                new_pos = []
                for m in range(len(xml_sets[k])):
                    # print(xml_sets[k][i][1])
                    p_x1 = int(xml_sets[k][m][1][0])
                    p_y1 = int(xml_sets[k][m][1][1])
                    p_x2 = int(xml_sets[k][m][1][2])
                    p_y2 = int(xml_sets[k][m][1][3])
                    n_x1 = max(0,int(p_x1*scales[j]))
                    n_y1 = max(0,int(p_y1*scales[j]))
                    n_x2 = int(p_x2*scales[j])
                    n_y2 = int(p_y2*scales[j])
                    if debug >= 2: print('Ori:',p_x1,p_y1,p_x2,p_y2, 'New:', n_x1,n_y1,n_x2,n_y2)
                    # cv2.rectangle(scaled_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                    new_pos.append([n_x1,n_y1,n_x2,n_y2])

            if not(noise[0]==0 and noise[1]==0): # Add noise
                for ratio in range(noise[0],noise[1],iter):
                    mean = 0
                    var = 0.05 #0.00001  # 0.5
                    sigma = var ** 0.5
                    if grey_img: ch = 1
                    gauss = np.random.normal(mean, sigma, (row, col, ch))
                    gauss = gauss.reshape(row, col, ch)
                    noisy_img = scaled_img + gauss * ratio
                    new_name = out_dir + '/' + ori_file+'_S%02d_N%02d.jpg' % (j,ratio)
                    cv2.imwrite(new_name, noisy_img)
                    if using_xml==1:
                        xml_name = ori_file+'_S%02d_N%02d' % (j,ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k,xml_name,new_pos)
                    cnt += 1
            if not (light[0] == 0 and light[1] == 0):  # Add Dark
                for ratio in range(light[0],light[1],iter):
                    invGamma = 1.0+ ratio*0.01
                    table = np.array([((i / 255.0) ** invGamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")
                    gaussian_img = cv2.LUT(scaled_img, table)

                    new_name = out_dir + '/' + ori_file+'_S%02d_L%02d.jpg' % (j,ratio)
                    cv2.imwrite(new_name, gaussian_img)
                    if using_xml==1:
                        xml_name = ori_file+'_S%02d_L%02d' % (j,ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k,xml_name,new_pos)
                    cnt += 1
            if (noise[0]==0 and noise[1]==0 and light[0] == 0 and light[1] == 0):
                new_name = out_dir + '/' + ori_file+'_S%02d.jpg' % j
                cv2.imwrite(new_name, scaled_img)
                if using_xml == 1:
                    xml_name = ori_file+'_S%02d' % j
                    # if debug >= 2: print(new_pos)
                    Save_XML(k,xml_name,new_pos)
                cnt+=1

    return cnt

def flip_images(noise=[0,90], light=[0,100], rate=4): #4 6 8 10
    cnt=0
    X_scale_data = []
    scales=[1,2,3,3]
    iter = int(40 - (rate*5))
    if rate<=0: return 0

    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        # img_data = cv2.resize(img_data, (basewidth, hsize))
        # img_data =  #X_imgs[k]
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        # w_y, w_x,_ = X_imgs[k].shape
        # print(w_x, w_y)
        if using_xml == 1:
            if debug >= 2:
                print(xml_sets[k])
                print(len(xml_sets[k]))
            for i in range(len(xml_sets[k])):
                if debug >= 2:
                    print(xml_sets[k][i][0])
                    print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)

        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)
        for j in range(scales[rate-1]):
            # w_y, w_x, w_c = img_data.shape
            if j==0: #up down flip
                scaled_img = cv2.flip( img_data, 0 )
            elif j==1: #left right flip
                scaled_img = cv2.flip( img_data, 1 )
            else:
                scaled_img = cv2.flip(img_data, -1)
            row, col, ch = scaled_img.shape

            if using_xml == 1:
                #[보류] xml 좌표 계산 다시 해야 함 ,
                new_pos = []
                for m in range(len(xml_sets[k])):
                    # print(xml_sets[k][i][1])
                    p_x1 = int(xml_sets[k][m][1][0])
                    p_y1 = int(xml_sets[k][m][1][1])
                    p_x2 = int(xml_sets[k][m][1][2])
                    p_y2 = int(xml_sets[k][m][1][3])
                    n_x1 = max(0,int(p_x1*scales[j]))
                    n_y1 = max(0,int(p_y1*scales[j]))
                    n_x2 = int(p_x2*scales[j])
                    n_y2 = int(p_y2*scales[j])
                    if debug >= 2: print('Ori:',p_x1,p_y1,p_x2,p_y2, 'New:', n_x1,n_y1,n_x2,n_y2)
                    # cv2.rectangle(scaled_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                    new_pos.append([n_x1,n_y1,n_x2,n_y2])

            if not(noise[0]==0 and noise[1]==0): # Add noise
                for ratio in range(noise[0],noise[1],iter):
                    mean = 0
                    var = 0.05 #0.00001  # 0.5
                    sigma = var ** 0.5
                    if grey_img: ch = 1
                    gauss = np.random.normal(mean, sigma, (row, col, ch))
                    gauss = gauss.reshape(row, col, ch)
                    noisy_img = scaled_img + gauss * ratio
                    new_name = out_dir + '/' + ori_file+'_F%02d_N%02d.jpg' % (j,ratio)
                    cv2.imwrite(new_name, noisy_img)
                    if using_xml==1:
                        xml_name = ori_file+'_F%02d_N%02d' % (j,ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k,xml_name,new_pos)
                    cnt += 1
            if not (light[0] == 0 and light[1] == 0):  # Add Dark
                for ratio in range(light[0],light[1],iter):
                    invGamma = 1.0+ ratio*0.01
                    table = np.array([((i / 255.0) ** invGamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")
                    gaussian_img = cv2.LUT(scaled_img, table)

                    new_name = out_dir + '/' + ori_file+'_F%02d_L%02d.jpg' % (j,ratio)
                    cv2.imwrite(new_name, gaussian_img)
                    if using_xml==1:
                        xml_name = ori_file+'_F%02d_L%02d' % (j,ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k,xml_name,new_pos)
                    cnt += 1
            if (noise[0]==0 and noise[1]==0 and light[0] == 0 and light[1] == 0):
                new_name = out_dir + '/' + ori_file+'_F%02d.jpg' % j
                cv2.imwrite(new_name, scaled_img)
                if using_xml == 1:
                    xml_name = ori_file+'_F%02d' % j
                    # if debug >= 2: print(new_pos)
                    Save_XML(k,xml_name,new_pos)
                cnt+=1

    return cnt

def noise_images(rate=4):
    noise = [30, 100]
    if debug >= 2: print(scale_list, len(scale_list))
    cnt=0
    # Various settings needed for Tensorflow operation
    scales =[]
    X_scale_data = []
    if rate==1 :
        iter = 55
        noise[0] = 50
    elif rate == 2:
        iter = 40
        noise[0] = 40
    elif rate == 3:
        iter = 30
    elif rate == 4:
        iter = 20
    else :
        return 0

    for index in range(noise[0],noise[1], iter):
        scales.append(index )

    if debug >= 2: print(scales, len(scales))

    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        row, col, ch = img_data.shape
        # img_data = cv2.resize(img_data, (basewidth, hsize))
        # img_data =  #X_imgs[k]
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        # w_y, w_x,_ = X_imgs[k].shape
        # print(w_x, w_y)
        if using_xml == 1:
            if debug >= 2:
                print(xml_sets[k])
                print(len(xml_sets[k]))
            for i in range(len(xml_sets[k])):
                if debug >= 2:
                    print(xml_sets[k][i][0])
                    print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)
        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)
        for j in range(len(scales)):
            mean = 0
            var = 0.05  # 0.00001  # 0.5
            sigma = var ** 0.5
            if grey_img: ch = 1
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy_img = img_data + gauss * scales[j]
            if using_xml == 1:
                new_pos = []
                for m in range(len(xml_sets[k])):
                    # print(xml_sets[k][i][1])
                    p_x1 = int(xml_sets[k][m][1][0])
                    p_y1 = int(xml_sets[k][m][1][1])
                    p_x2 = int(xml_sets[k][m][1][2])
                    p_y2 = int(xml_sets[k][m][1][3])
                    n_x1 = p_x1
                    n_y1 = p_y1
                    n_x2 = p_x2
                    n_y2 = p_y2
                    if debug >= 2: print('Ori:',p_x1,p_y1,p_x2,p_y2, 'New:', n_x1,n_y1,n_x2,n_y2)
                    # cv2.rectangle(noisy_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                    new_pos.append([n_x1,n_y1,n_x2,n_y2])

            new_name = out_dir + '/' + ori_file+'_N%01d.jpg' % j
            cv2.imwrite(new_name, noisy_img)
            if using_xml == 1:
                xml_name = ori_file+'_N%01d' % j
                # if debug >= 2: print(new_pos)
                Save_XML(k,xml_name,new_pos)
            cnt+=1

    return cnt

def light_images(rate=4):
    light = [30, 100]
    if debug >= 2: print(scale_list, len(scale_list))
    cnt=0
    # Various settings needed for Tensorflow operation
    scales =[]
    X_scale_data = []
    if rate==1 :
        iter = 55
        light[0] = 50
    elif rate == 2:
        iter = 40
        light[0] = 40
    elif rate == 3:
        iter = 30
    elif rate == 4:
        iter = 20
    else:
        return 0
    for index in range(light[0],light[1], iter):
        scales.append(index )

    if debug >= 2: print(scales, len(scales))

    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        row, col, ch = img_data.shape
        # img_data = cv2.resize(img_data, (basewidth, hsize))
        # img_data =  #X_imgs[k]
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        # w_y, w_x,_ = X_imgs[k].shape
        # print(w_x, w_y)
        if using_xml == 1:
            if debug >= 2:
                print(xml_sets[k])
                print(len(xml_sets[k]))
            for i in range(len(xml_sets[k])):
                if debug >= 2:
                    print(xml_sets[k][i][0])
                    print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)
        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)
        for j in range(len(scales)):
            invGamma = 1.0 + scales[j] * 0.01
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            gaussian_img = cv2.LUT(img_data, table)
            if using_xml == 1:
                new_pos = []
                for m in range(len(xml_sets[k])):
                    # print(xml_sets[k][i][1])
                    p_x1 = int(xml_sets[k][m][1][0])
                    p_y1 = int(xml_sets[k][m][1][1])
                    p_x2 = int(xml_sets[k][m][1][2])
                    p_y2 = int(xml_sets[k][m][1][3])
                    n_x1 = p_x1
                    n_y1 = p_y1
                    n_x2 = p_x2
                    n_y2 = p_y2
                    if debug >= 2: print('Ori:',p_x1,p_y1,p_x2,p_y2, 'New:', n_x1,n_y1,n_x2,n_y2)
                    # cv2.rectangle(gaussian_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                    new_pos.append([n_x1,n_y1,n_x2,n_y2])

            new_name = out_dir + '/' + ori_file+'_L%01d.jpg' % j
            cv2.imwrite(new_name, gaussian_img)
            if using_xml == 1:
                xml_name = ori_file+'_L%01d' % j
                # if debug >= 2: print(new_pos)
                Save_XML(k,xml_name,new_pos)
            cnt+=1

    return cnt

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_images(angle=2, noise=[0,90],light=[0,100], rate=4):
    cnt=0
    v_angle=[]
    interval = 25
    for i in range(-angle*100,(angle)*100+interval,interval):
        if i!=0: v_angle.append(i/100.)
    if debug >= 2: print(v_angle, len(v_angle))
    iter = int(40 - (rate * 5))
    if rate<=4:
        count= 24-(5-rate)*4
        vv_angle = random.sample(v_angle, count)
        v_angle=[]
        for i in range(count):
            v_angle.append(vv_angle[i])
        v_angle.sort()
    else:
        return 0
    if debug >= 2: print(v_angle, len(v_angle))

    # if rate == 4:
    #     interval = 25
    #     iter = 20
    # elif rate == 3:
    #     interval = 35
    #     iter = 25
    # elif rate == 2:
    #     interval = 45
    #     iter = 30
    # elif rate == 1:
    #     interval = 75
    #     iter = 40
    # else :
    #     return 0


    X_scale_data = []
    for k in range(len(X_img_paths)):
        print(X_img_paths[k])
        img_data = cv2.imread(X_img_paths[k])
        ori_file = os.path.splitext(os.path.basename(X_img_paths[k]))[0]
        if using_xml == 1:
            if debug >= 2:
                print(xml_sets[k])
                print(len(xml_sets[k]))
            for i in range(len(xml_sets[k])):
                if debug >= 2:
                    print(xml_sets[k][i][0])
                    print(xml_sets[k][i][1])
                p_x1 = int(xml_sets[k][i][1][0])
                p_y1 = int(xml_sets[k][i][1][1])
                p_x2 = int(xml_sets[k][i][1][2])
                p_y2 = int(xml_sets[k][i][1][3])
                # cv2.rectangle(img_data, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 1)
        # test
        # rname = out_dir + '/' + '_%010d.jpg' % (k)
        # cv2.imwrite(rname, img_data)
        for j in range(len(v_angle)):
            # scaled_img = cv2.resize(img_data, (0, 0), fx=scales[j], fy=scales[j])
            scaled_img = rotateImage(img_data, v_angle[j])
            if grey_img: ch = 1
            row, col, ch = scaled_img.shape
            if using_xml == 1:
                new_pos = []
                for m in range(len(xml_sets[k])):
                    # print(xml_sets[k][i][1])
                    p_x1 = int(xml_sets[k][m][1][0])
                    p_y1 = int(xml_sets[k][m][1][1])
                    p_x2 = int(xml_sets[k][m][1][2])
                    p_y2 = int(xml_sets[k][m][1][3])
                    # n_x1 = max(0, p_x1)
                    # n_y1 = max(0, p_y1)
                    # n_x2 = p_x2
                    # n_y2 = p_y2
                    # print(p_x2-p_x1,p_y2-p_y1)
                    ratio = (p_y2-p_y1)/ (p_x2-p_x1) *2
                    # print(ratio)
                    # print(abs(v_angle[j]), abs(v_angle[j])*2, abs(v_angle[j])*3, abs(v_angle[j])*6)
                    # print(abs(v_angle[j])*ratio, abs(v_angle[j])*2*ratio, abs(v_angle[j])*3*ratio, abs(v_angle[j])*6*ratio)
                    if int(len(v_angle)/2) <= j : #counter clockwise turn
                        n_x1 = max(0,int(p_x1 + abs(v_angle[j])*1*ratio))
                        n_y1 = max(0,int(p_y1 - abs(v_angle[j])*2*ratio))
                        n_x2 = int(p_x2 + abs(v_angle[j])*7*ratio)
                        n_y2 = int(p_y2+ abs(v_angle[j])*2*ratio)
                    else: #clockwise turn
                        n_x1 = max(0, int(p_x1 - abs(v_angle[j])*7*ratio))
                        n_y1 = max(0,int(p_y1 - abs(v_angle[j])*2*ratio))
                        n_x2 = int(p_x2 - abs(v_angle[j])*1*ratio)
                        n_y2 = int(p_y2 + abs(v_angle[j])*2*ratio)
                    if debug >= 2: print('Ori:',p_x1,p_y1,p_x2,p_y2, 'New:', n_x1,n_y1,n_x2,n_y2)
                    # cv2.rectangle(scaled_img, (n_x1, n_y1), (n_x2, n_y2), (0, 255, 0), 1)
                    new_pos.append([n_x1,n_y1,n_x2,n_y2])

            if not (noise[0] == 0 and noise[1] == 0):  # Add noise
                for ratio in range(noise[0], noise[1], iter):
                    mean = 0
                    var = 0.05  # 0.00001  # 0.5
                    sigma = var ** 0.5
                    if grey_img: ch = 1
                    gauss = np.random.normal(mean, sigma, (row, col, ch))
                    gauss = gauss.reshape(row, col, ch)
                    noisy_img = scaled_img + gauss * ratio
                    new_name = out_dir + '/' + ori_file + '_R%02d_N%02d.jpg' % ( j, ratio)
                    cv2.imwrite(new_name, noisy_img)
                    if using_xml == 1:
                        xml_name = ori_file + '_R%02d_N%02d' % ( j, ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k, xml_name, new_pos)
                    cnt += 1
            if not (light[0] == 0 and light[1] == 0):  # Add Dark
                for ratio in range(light[0], light[1], iter):
                    invGamma = 1.0 + ratio * 0.01
                    table = np.array([((i / 255.0) ** invGamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")
                    gaussian_img = cv2.LUT(scaled_img, table)

                    new_name = out_dir + '/' + ori_file + '_R%02d_L%02d.jpg' % ( j, ratio)
                    cv2.imwrite(new_name, gaussian_img)
                    if using_xml == 1:
                        xml_name = ori_file + '_R%02d_L%02d' % ( j, ratio)
                        # if debug >= 2: print(new_pos)
                        Save_XML(k, xml_name, new_pos)
                    cnt += 1
            if (noise[0] == 0 and noise[1] == 0 and light[0] == 0 and light[1] == 0):
                new_name = out_dir + '/' + ori_file+'_R%02d.jpg' % j
                cv2.imwrite(new_name, scaled_img)
                if using_xml == 1:
                    xml_name = ori_file+'_R%02d' % j
                    # if debug >= 2: print(new_pos)
                    Save_XML(k,xml_name,new_pos)
                cnt+=1

    return cnt

#========================================================
# Main
#========================================================
debug =0 # 2 #0 final    #1 거의 결과  #2 개발중
mode = 1 # 0: predict count  1: make real data
cfg_fname = 'pet_augmentation_xml.cfg' #cfg 파일이 없으면 하드코딩된 설정값 이용, # cfg 파일이 있으면 파일 내의 설정값 사용
using_xml =0 # 0: no xml  1: using xml 을 읽어 새로운 이미지마다  생성(세그멘테이션 있을 때)
wantednum=0 #생성되기 원하는 파일 수,  특별히 원하지 않으면 0, 100으로 지정시 -생성이 500개 되면 400개를 삭제하고 100개만 남김
usingfilenum=0 #파일 개수 모를때 0 , 만일 100개중 50개만 사용하고 싶으면 50
grey_img = 1 # 영상이 단색이면 1, 컬러면 0
data_dir = '/home/turbo/share/sda1/data/origin/1gr'  #소스 이미지 폴더
out_dir = '/home/turbo/share/sda1/data/origin/out' #새로 생생된 이미지 저장할 폴더
if not os.path.isdir(out_dir): os.mkdir(out_dir)

make_option=[0,1,0,0,0,0,0] #0번째는 무시 , 0이상 숫자면 셋팅된 걸로
# make_option=[0,0,0,0,0,0] #아무 것도 선택 안됨
# make_option=[0,1,0,0,0,0,0] #이미지 크기 조절 resize   (noise_option, light_option으로 추가 가능)
# make_option=[0,0,2,0,0,0,0] #이미지 일정 부분 자르기 crop (noise_option, light_option으로 추가 가능)
# make_option=[0,0,0,3,0,0,0] #이미지 회전 ratate (noise_option, light_option으로 추가 가능)
# make_option=[0,0,0,0,4,0,0] #이미지 flip (noise_option, light_option으로 추가 가능)
# make_option=[0,0,0,0,0,5,0] #잡음 noise  (단순히 잡음만 추가)
# make_option=[0,0,0,0,0,0,6] #밝기 light  (단순히 밝기만 추가)
# make_option=[0,1,2,3,4,5,6]  #모두 선택
make_option_sub=[
    [0, 0, 0, 0, 0, 0, 0], #0   0번째는 무시
    [0, 4, 0, 0, 0, 0, 0], #1   1번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
    [0, 0, 4, 0, 0, 0, 0], #2   2번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
    [0, 0, 0, 4, 0, 0, 0], #3   3번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
    [0, 0, 0, 0, 4, 0, 0], #4   4번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
    [0, 0, 0, 0, 0, 4, 0], #5   5번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
    [0, 0, 0, 0, 0, 0, 4]  #6   6번째만 의미 있음(나머지는 0) : 1,2,3,4에 따라 rate가 조정되어 생성되는 파일 개수가 늘어남
]
noise_option=[
    [0,0],  #0번째는 무시
    [0,90], #1 make_option 1에 적용 : [0,0]이면 noise 포함 안함,  [0,90]:0-90% 사이의  noise 적용
    [0,90], #2 make_option 2에 적용 : [0,0]이면 noise 포함 안함,  [0,90]:0-90% 사이의  noise 적용
    [0,90], #3 make_option 3에 적용 : [0,0]이면 noise 포함 안함,  [0,90]:0-90% 사이의  noise 적용
    [0,90]  #4 make_option 4에 적용 : [0,0]이면 noise 포함 안함,  [0,90]:0-90% 사이의  noise 적용
]
light_option=[
    [0,0],  #0번째는 무시
    [0,100],  #1 make_option 1에 적용 : [0,0]이면 light 포함 안함,  [0,100]:0-100% 사이의  light 적용
    [0,100],  #2 make_option 1에 적용 : [0,0]이면 light 포함 안함,  [0,100]:0-100% 사이의  light 적용
    [0,100],  #3 make_option 1에 적용 : [0,0]이면 light 포함 안함,  [0,100]:0-100% 사이의  light 적용
    [0,100]   #4 make_option 1에 적용 : [0,0]이면 light 포함 안함,  [0,100]:0-100% 사이의  light 적용
]

#아래 두표는 생셩될 파일 갯수 계산용임(수정 X)
count_table1=[
    [ 0,  0,  0,  0], #0번째는 무시
    [ 2,  4,  6,  8], #make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수
    [ 8, 16, 24, 32], #make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수
    [ 8, 12, 16, 20], #make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수
    [ 1,  2,  3,  3], # make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수
    [ 1,  2,  3,  4], #5번째는 무시
    [ 1,  2,  3,  4]  #6번째는 무시
]
count_table2=[
    [ 3,  3,  4,  5], #make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수, noise_option이 0이 아닐 때
    [ 3,  4,  4,  5]  #make_option_sub의 단계 1,2,3,4에 따른 파일 생성 갯수, light_option이 0이 아닐 때
]

#======================================================
# cfg 파일 읽어 설정하기
#======================================================
target_files=[]
if os.path.exists(cfg_fname):
    print("Read cfg file and setting options.")
    lines = [line.rstrip('\n') for line in open(cfg_fname)]
    for i in  range(len(lines)):
        if lines[i].find('<mode>') != -1:
            i+=1
            if (i - 2) >= len(lines): break
            mode=int(lines[i])
            # print('mode: ', mode)
        elif lines[i].find('<xml>') != -1:
            i+=1
            if (i - 2) >= len(lines): break
            using_xml=int(lines[i])
            # print('xml: ', using_xml)
        elif lines[i].find('<data_dir>') != -1:
            i += 1
            if (i - 2) >= len(lines): break
            data_dir = lines[i]
            print('data_dir: ', data_dir)
        elif lines[i].find('<out_dir>') != -1:
            i+=1
            if (i - 2) >= len(lines): break
            out_dir=lines[i]
            if not os.path.isdir(out_dir): os.mkdir(out_dir)
            print('out_dir: ', out_dir)
        elif lines[i].find('<target_outfile_num>') != -1:
            i+=1
            if (i - 2) >= len(lines): break
            wantednum=int(lines[i])
            print('target_outfile_num: ', wantednum)
        elif lines[i].find('<options>') != -1:
            while True:
                i+=1
                if (len(lines[i])==0): i+=1
                if (i-2)>=len(lines): break
                if lines[i][0]=='<': break
                opt = lines[i].split(' ')
                make_option[int(opt[0])]=int(opt[1])
                make_option_sub[int(opt[0])][int(opt[0])] = int(opt[2])
                if int(opt[0])<=4:  #noise,light이저의 갯수로 조정해줘야 함
                    noise_option[int(opt[0])][0] = int(opt[3])
                    noise_option[int(opt[0])][1] = int(opt[4])
                    light_option[int(opt[0])][0] = int(opt[5])
                    light_option[int(opt[0])][1] = int(opt[6])

        elif lines[i].find('<files>') != -1:
            # print('f----')
            while True:
                i+=1
                # print('  ',i, len(lines),'  ')
                if (len(lines[i])==0): i+=1
                if (i+1)>len(lines): break
                if lines[i][0]=='<': break
                target_files.append(lines[i])
                # print(lines[i])
        elif lines[i].find('<using_file_num>') != -1:
            i+=1
            if (i - 2) >= len(lines): break
            # print(lines[i], end='')
            usingfilenum=int(lines[i])
            print('using_file_num: ', usingfilenum)
else:
    print("Can't read cfg file and use hard coding options.")
#======================================================
# 설정값 출력
#======================================================
print('mode: ', mode)
print('xml: ', using_xml)
print('make_option: ', make_option)
print('make_option_sub: ')
for i in range(len(make_option_sub)):
    print( make_option_sub[i])
print('noise_option: ')
for i in range(len(noise_option)):
    print(noise_option[i])
print('light_option: ')
for i in range(len(light_option)):
    print(light_option[i])

if os.path.exists(cfg_fname):
    print('target_files:  #', len(target_files))
    for i in range(len(target_files)):
        print(target_files[i])
else:
    ori_files=[]
    o_files = os.listdir(data_dir)
    for i in range(len(o_files)):
        d = os.path.splitext(os.path.basename(o_files[i]))[1]
        if d=='.jpg' or d=='.JPG':
            ori_files.append(o_files[i])
    print('target_files:  #', len(ori_files))
    for i in range(len(ori_files)):
        print(ori_files[i])

#======================================================
# 생성될 파일 수 계산   mode=0이면 파일 생성 안하고 계산만
#======================================================
total_pre_sum=0
for i in range(len(make_option_sub)-1):
    base_num =0
    pre_sum = 0
    # print(make_option_sub[i+1][i+1])
    if make_option_sub[i+1][i+1]!=0:
        # print( count_table1[i+1][make_option_sub[i+1]])
        # print(make_option_sub[i+1][i+1])
        pre_sum=0
        base_num= count_table1[i+1][make_option_sub[i+1][i+1]-1] * make_option[i]
        if make_option[i+1] > 0 : temp_a=1
        else : temp_a=0
        base_num= count_table1[i+1][make_option_sub[i+1][i+1]-1] * temp_a
        # print(i+1,'  base_num: ',base_num)
        if (i+1)<=3 and noise_option[i+1][1]!=0:
            # print(i + 1, '  pre_sum(noise): ', base_num * count_table2[0][make_option_sub[i+1][i+1]-1])
            pre_sum += base_num * count_table2[0][make_option_sub[i+1][i+1]-1]
        else:
            pre_sum =base_num  #base_num
            # print(i + 1, '  pre_sum(noise): ',0)
        if (i+1)<=3 and light_option[i+1][1]!=0:
            # print(i + 1, '  pre_sum(light): ', base_num * count_table2[1][make_option_sub[i+1][i+1]-1])
            pre_sum += base_num * count_table2[1][make_option_sub[i+1][i+1]-1]
        else:
            pre_sum = base_num #base_num
            # print(i + 1, '  pre_sum(light): ', 0)
        print(i + 1, '[sum]: ',pre_sum )
    total_pre_sum+=pre_sum

if os.path.exists(cfg_fname):
    if usingfilenum==0: usingfilenum=len(target_files)
    if usingfilenum > len(target_files): usingfilenum=len(target_files)
else:
    # ori_files = os.listdir(data_dir)
    if usingfilenum==0: usingfilenum=len(ori_files)
    if usingfilenum > len(target_files): usingfilenum=len(ori_files)
print('Original target file_num: ',usingfilenum)

if usingfilenum < len(target_files):
    delnum = len(target_files)-usingfilenum
    rand_smpl1 = [ i for i in sorted(random.sample(range(len(target_files)), delnum))]
    # print(rand_smpl1)
    for i in range(len(rand_smpl1)-1,-1,-1):
        # print(rand_smpl1[i])
        del target_files[i]
    # print(len(target_files))

print('Using file_num: ',usingfilenum)
print('predicted num per image: ',total_pre_sum,'  target_files #: ',usingfilenum)
total_pre_sums = total_pre_sum*usingfilenum
print('total predicted num: ',total_pre_sums)
if wantednum == 0:
    wantednum = total_pre_sums
delnum = total_pre_sums - wantednum
if delnum<0  : delnum=0
print('predicted del num: ',delnum)
if wantednum > total_pre_sums: wantednum = total_pre_sums
print('predicted Final num: ',wantednum)
if mode == 0: sys.exit()

#======================================================
# 이미지 소스 폴더 읽기
#======================================================
X_img_paths, xml_sets = get_image_paths(data_dir)
# if debug>=2 : print(X_img_paths)

scale_list = []
num_files = len(X_img_paths)
result_cnt = 0
start_num=1
create_cnt=0
Min_scale, Max_scale = 60, 95 #make_option 1 (이미지 크기 조절)에 사용, 60 - 95% 로 크기 조절

#======================================================
#1) Scaling  : 이미지 크기 조절
#======================================================
if make_option[1]!=0:
    Min_scale, Max_scale = 88, 110  # make_option 1 (이미지 크기 조절)에 사용, 88 - 110% 로 크기 조절
    pos=1
    create_cnt = central_scale_images(noise=noise_option[pos],light=light_option[pos], rate=make_option_sub[pos][1])
    # rate별 생성 데이타 수
    #  1   2    3   4  - rate
    #  2   4    6   8  - basic
    #  6  12   24  40  - noise, no light 3 3 4 5
    #  6  16   24  40  - no noise, light 3 4 4 5
    # 12  28   48  80  - nose + light
    start_num += create_cnt
    result_cnt += create_cnt
    print('1) create_cnt:',create_cnt)
#======================================================
# 2)Translation : 잘라내기 ,crop
#======================================================
if make_option[2]!=0:
    Min_scale, Max_scale = 96, 100  # make_option 2 (잘라낼 이미지 portion 조절)에 사용, 96 - 100% 로 크기 조절
    pos=2
    create_cnt = translate_images(noise=noise_option[pos],light=light_option[pos], rate=make_option_sub[pos][2])
    # rate별 생성 데이타 수
    #   1    2    3    4  - rate
    #   8   16   24   32  - basic
    #  24   48   96  160  - noise, no light 3 3 4 5
    #  24   64   96  160  - no noise, light 3 4 4 5
    #  48  112  192  320  - nose + light
    start_num += create_cnt
    result_cnt += create_cnt
    print('2) create_cnt:', create_cnt)
#======================================================
# 3)Rotation (at finer angles)  : 회전
#======================================================
if make_option[3]!=0:
    pos=3
    # make_option 3 (각도 조절에 사용, -10 -  10도로 크기 조절
    create_cnt = rotate_images(angle= 10, noise=noise_option[pos],light=light_option[pos], rate=make_option_sub[pos][3])
    # rate별 생성 데이타 수
    #   1   2    3    4  - rate
    #   8  12   16   20  - no noise, light
    #  24  36   64  100  - noise, no light 3 3 4 5
    #  24  48   64  100  - no noise, light 3 4 4 5
    #  48  84  128  200  - nose + light

    start_num += create_cnt
    result_cnt += create_cnt
    print('3) create_cnt:', create_cnt)

#======================================================
#4) Flip  : 상하,좌우 반전 이미지
#======================================================
if make_option[4]!=0:
    pos=4
    # make_option 4 (이미지 반전, rate: 1: 상하, 2:좌우 3,4:상하좌우 동시)
    create_cnt = flip_images(noise=noise_option[pos],light=light_option[pos], rate=make_option_sub[pos][4])
    # rate별 생성 데이타 수
    #  1   2    3   4  - rate
    #  1   2    3   3  - basic
    #  3   6   12  15  - noise, no light 3 3 4 5
    #  3   8   12  15  - no noise, light 3 4 4 5
    #  6  14   24  30  - nose + light
    start_num += create_cnt
    result_cnt += create_cnt
    print('4) create_cnt:',create_cnt)

#======================================================
# 5)Noise : 단순 잡음 추가
#======================================================
if make_option[5]!=0:
    pos=5
    create_cnt = noise_images(rate=make_option_sub[pos][5])
    # rate별 생성 데이타 수
    #   1   2    3    4  - rate
    #   1   2    3    4  - basic
    start_num += create_cnt
    result_cnt += create_cnt
    print('5) create_cnt:', create_cnt)

#======================================================
# 6)light : 단순 밝기 조절
#======================================================
if make_option[6]!=0:
    pos=6
    create_cnt = light_images(rate=make_option_sub[pos][6])
    # rate별 생성 데이타 수
    #   1   2    3    4  - rate
    #   1   2    3    4  - basic
    start_num += create_cnt
    result_cnt += create_cnt
    print('6) create_cnt:', create_cnt)

#======================================================
# 통계
#======================================================
print('Total_cnt:',result_cnt)

ori_files=[]
result_files=[]
ori_files = os.listdir(out_dir)
for i in range(len(ori_files)):
    x = os.path.splitext(os.path.basename(ori_files[i]))[1]
    if x=='.jpg' or x=='.JPG':
        result_files.append(ori_files[i])
result_files.sort()
print('Current_cnt:',len(result_files))

delnum=total_pre_sums-wantednum
print(delnum)
if delnum >= len(result_files): delnum=0
if delnum <= 0: exit()
rand_smpl = [ result_files[i] for i in sorted(random.sample(range(len(result_files)), delnum)) ]
# print(rand_smpl)
for i in range(len(rand_smpl)):
    x = os.path.splitext(os.path.basename(rand_smpl[i]))[0]
    os.remove(out_dir + '/' + x + '.jpg')
    if using_xml == 1:
        os.remove(out_dir + '/' + x + '.xml')
ori_files = os.listdir(out_dir)
if using_xml == 1:
    print('Final_cnt:',len(ori_files)/2)
else: print('Final_cnt:',len(ori_files))

