import os

source_dir  = '/home/turbo/share/sda1/data/simple'
save_dir  = '/home/turbo/share/sda1/data/simple_divide'
if not os.path.isdir(save_dir): os.mkdir(save_dir)
# train 디렉토리와 test 디렉토리를 정의합니다.
train_dir = save_dir+'/train'
test_dir  = save_dir+'/valid'
test_portion = 0.2   #train:test 분배비율 - test 쪽 값만 써주면 됨(합이 1이 되게)

def divide_data(test_portion):
    import random
    import copy
    import shutil

    if not os.path.isdir(train_dir): shutil.copytree(source_dir, train_dir)
    if not os.path.isdir(test_dir): os.mkdir(test_dir)

    sub_dir = os.listdir(source_dir)
    for i in range(len(sub_dir)):
        out_dir1 = train_dir + '/' + sub_dir[i]
        out_dir2 = test_dir + '/' + sub_dir[i]
        if not os.path.isdir(out_dir1): os.mkdir(out_dir1)
        if not os.path.isdir(out_dir2): os.mkdir(out_dir2)
        ori_files = []
        result_files = []
        ori_files = os.listdir(source_dir+'/'+sub_dir[i])
        for j in range(len(ori_files)):
            x = os.path.splitext(os.path.basename(ori_files[j]))[1]
            if x == '.jpg' or x == '.JPG':
                result_files.append(ori_files[j])
        result_files.sort()
        print('Current_cnt:', len(result_files))

        move_num = int(len(result_files) * test_portion)
        print('move_num :',move_num)
        if len(result_files) == 0 : exit()
        if move_num >= len(result_files): move_num = 0
        if move_num <= 0: exit()
        ori_files = os.listdir(out_dir2)
        if len(ori_files) >= move_num: continue
        rand_smpl = [result_files[i] for i in sorted(random.sample(range(len(result_files)), move_num))]
        # print(rand_smpl)
        for j in range(len(rand_smpl)):
            x = os.path.splitext(os.path.basename(rand_smpl[j]))[0]
            # os.remove(out_dir2 + '/' + x + '.jpg')
            shutil.move(out_dir1 + '/'+x + '.jpg', out_dir2 + '/')
        ori_files = os.listdir(out_dir1)
        print(sub_dir[i]+'_train_dir_cnt:', len(ori_files))
        ori_files = os.listdir(out_dir2)
        print(sub_dir[i]+'_test_dir_cnt:', len(ori_files))

# train 디렉토리가 없으면 만들고 정해진 비율로 tarin과 test를 분배한다.
if not os.path.isdir(train_dir): divide_data(test_portion)
