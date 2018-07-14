# 필요한 모듈을 임포트합니다.
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys

# 케라스 관련한 필요한 모듈 임포트
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

# 싸이킷 럿 관련 모듈 업로드
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight

save_model='/home/turbo/study/playground/pet_resnet50/model/vgg16_d0.h5'

# ResNet50 으로 케라스에서 ResNet 모형을 불러옵니다
# model = ResNet50(include_top=True, weights='imagenet')
# input_shape=model.layers[0].output_shape[1:3]
# print(input_shape)

# vgg16 으로 케라스에서 vgg16t 모형을 불러옵니다
model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
input_shape=model.layers[0].output_shape[1:3]
print(input_shape)


# 데이터 어그멘테이션을 해 줍니다.
# rescale 1./128은 이미지 픽셀을 128로 나눠 주는 기능을 합니다
# rotation_range=20은 20도 범위로 사진을 회전해줍니다.
# brightness_range는 사진의 밝기를 1.0배에서 2.5배만큼 회전해줍니다.
# horizontal_flip 은 사진을 좌 우 바꿔줍니다.
# vertical_flip은 사진을 위 아래로 뒤집어줍니다.
datagen_train = ImageDataGenerator(rescale=1./128)
# datagen_train = ImageDataGenerator(
#       rescale=1./128,
#       rotation_range=20,
#       brightness_range=(1.0,2.5),
#       horizontal_flip=True,
#       vertical_flip=True)

# 테스트 용으로 쓸 이미지는 픽셀만 128배로 줄여줍니다.
datagen_test = ImageDataGenerator(rescale=1./128)

# train 디렉토리와 test 디렉토리를 정의합니다.
train_dir = '/home/turbo/share/sda1/data/train7'
test_dir  = '/home/turbo/share/sda1/data/valid7'
test_portion = 0.2   #train:test 분배비율 - test 쪽 값만 써주면 됨(합이 1이 되게)

def divide_data(test_portion):
    import random
    import copy
    import shutil

    source_dir  = '/home/turbo/share/sda1/data/original_augmented'

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
# if not os.path.isdir(train_dir): divide_data(test_portion)

train_files=[]
train_file = os.listdir(train_dir)
for i in range(len(train_file)):
    train_files += os.listdir(train_dir+'/'+train_file[i])
test_files=[]
test_file = os.listdir(test_dir)
for i in range(len(test_file)):
    test_files += os.listdir(test_dir+'/'+test_file[i])
print(len(train_files), len(test_files))

# 모델 훈련에 쓸 계수들을 입력해줍니다.
# steps_per_epoch_train은 한 에포크당 얼마나 트레인용 이미지를 훈련할지 알려줍니다.
# 배치크기를 20으로 정했으므로 한 에포크당 step 수는 전체 사진 수 451 개 에서 20으로 나눠야 겠지요?
# 그리고 정수 반환을 위해 451//20으로 해줍니다.
# steps_per_epoch_val도 위와 비슷합니다.
epochs=2
batch_size = 100
# steps_per_epoch_train=22500//100
# steps_per_epoch_val=7500//10

# epochs=2
# batch_size = 20
steps_per_epoch_train=len(train_files)//batch_size
steps_per_epoch_val=len(test_files)//batch_size
target_size=(224, 224)
# 훈련용 사진들을 불러옵니다.
# 위에 정의한 datagen_train에 flow_from_directory를 이용해서 케라스와 훈련용 사진들을 연결해 줍니다.
# 훈련 폴더는 train1이고, train1 안에 서브 폴더로 1gr, 2gr, 3gr이 있습니다.
# 이렇게 두면 케라스가 1gr, 2gr, 3gr 안의 사진을 읽게 됩니다.
# generator_train.reset()
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# generator_test.reset()
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

# 각 1gr 2gr 3gr의 사진들의 수는 서로 다릅니다. 이 불균형한 비율을 계산하기 위해 싸이킷런 compute_class_weight 모듈을 불러옵니다.
from sklearn.utils.class_weight import compute_class_weight

steps_test = generator_test.n / batch_size
print(steps_test)
# train의 클래스들을 불러옵니다. 1gr = 0, 2gr = 1, 3gr = 2 입니다.
cls_train = generator_train.classes

# test의 클래스를 불러옵니다.
cls_test = generator_test.classes

# 1gr 2gr 3gr의 사진수의 비율을 계산합니다.
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(cls_train), y=cls_train)

print(class_weight)

if not os.path.isfile(save_model):
    # ResNet 50 모형을 요약해서 보여줍니다.
    model.summary()

## !pip install -q pydot, !apt-get install graphviz -y 으로 pydot, graphviz 설치해야 함.
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# vgg16 모형을 그림으로 볼 수 있습니다.
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Transfer Learning을 위해 맨 마지막 부분의 Flatten층을 잘라서 불러옵니다.
# 맨 마지막층의 이름이 block5_pool이기 때문에 model.get_layer('block5_pool')를 해서 block5_pool의 텐서를 불러옵니다.
transfer_layer = model.get_layer('block5_pool')

print(transfer_layer.output)

# 새로운 모델을 정의합니다. Resnet50의 최초 인풋부터 flatten_1까지의 모델을 새로 만듭니다.
# 이렇게 하면 기존 ResNet50 모형을 인풋부터 Flatten까지 자를 수 있습니다.
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

# ResNet50의 모델의 인풋 크기를 확인합니다.
print(model.input)

# Transfer Learning에 쓰일 모델을 새로 정의합니다
# new_model.add(conv_model)을 이용하여 ResNet50 모형을
# 인풋부터 Flatten까지 자른 모델인 conv_model과 Transfer Learning에 붙일 새로운 모델인 new_model을 정의해줍니다.
# 이 끝단의 최적 모델을 찾아야 할 것 같습니다.
new_model = Sequential()
new_model.add(conv_model)
new_model.add(Flatten())
new_model.add(Dense(256, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(3, activation='softmax'))

# 밑에 모델은 augmentation이 없었을 시 잘 작동했습니다.(ImageDataGenerator에 rescale만 1./128로 해줌)
# new_model = Sequential()
# new_model.add(conv_model)
# new_model.add(Dense(1000, activation='relu'))
# new_model.add(Dropout(0.5))
# new_model.add(Dense(3), activation='softmax')

# new_model의 그림을 그려 봅니다.
# SVG(model_to_dot(new_model, show_shapes=True).create(prog='dot', format='svg'))

# conv_model의 계수 업데이트가 가능한지(trainable)를 보여줍니다. True면 계수 업데이트가 가능한 것이고, False면 계수 업데이트 없이 기존에 학습된 Pre-trained weight을 사용합니다
def print_layer_trainable():
  for layer in conv_model.layers:
    print("{0}: {1}".format(layer.trainable, layer.name))

# conv_model의 모든 층을 학습 불가능하게 얼립니다.
for layer in conv_model.layers:
  layer.trainable = False

print(len(conv_model.layers))

# flatten_2까지 trainable 상태를 False로 만들어 줍니다.
for layer in conv_model.layers[:]:
  layer.trainable=False

print_layer_trainable()


# 훈련에 쓸 loss 함수와, metrics를 정의합니다. 그냥 붙여넣기 해 주세요.
loss='categorical_crossentropy'
metrics=['categorical_accuracy']

# optimizer에 사용할 Adam 계수를 정의하고, 위에 정의한 loss와 metrics, 그리고 optimizer를 compile해 줍니다.
optimizer_fine = Adam(lr=1e-5)
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

# early stopping 콜백함수를 만들어서, val_loss를 관찰해보고 만약 10번 이상 계속 val_loss가 떨어질 시 훈련을 종료해줍니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10)

# 위에 정의한 steps_per_epoch_train에 50을 곱해서 이미지 뻥튀기한 사진들이 가능한 한 많이 반영되도록 합니다.
history1 = new_model.fit_generator(generator=generator_train,
                                  epochs=100,
                                  steps_per_epoch=steps_per_epoch_train, #*50,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_per_epoch_val,
                                  callbacks=[earlystopping])

# 학습 상태를 plot 해 봅니다.
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history1.history['loss'], 'y', label='train loss')
loss_ax.plot(history1.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history1.history['categorical_accuracy'], 'b', label='train acc')
acc_ax.plot(history1.history['val_categorical_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper right')
acc_ax.legend(loc='lower right')
plt.title("Alzheimer's disease pridiction on resnet 50 model")

plt.show()

# 모델을 원하는 위치에 저장해 줍니다.
# model.save('/home/turbo/study/playground/pet_resnet50/model/resnet50_3.h5')
model.save(save_model)

# 모델의 label을 확인합니다.
class_name = generator_train.class_indices.keys()

# path_join 함수를 만듭니다.
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

# 맨 위쪽에 정의한 generator_train에서 사진들의 실제 이미지를 불러와서 path를 만들어 줍니다.
image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())
print(class_names)


# 사진을 불러올 함수를 정의합니다. 그냥 쓰시면 됩니다.
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# confusion_matrix를 그려줄 함수를 정의합니다. 그냥 쓰시면 됩니다.
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.
    re = classification_report(y_true=cls_test,
                               y_pred=cls_pred)
    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)
    print(re)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# 최종적으로 위에 정의한 함수를 이용하여 잘못 분류한 사진들을 그려주고,
# confusion_matrix를 구해서 학습한 모델의 recall, precision, f1_score를 불러 옵니다.
def example_errors():
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.
    generator_test.reset()

    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examples of mis-classified images.
    plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

example_errors()

