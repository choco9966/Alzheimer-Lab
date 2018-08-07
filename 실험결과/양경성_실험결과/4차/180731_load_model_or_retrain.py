#저장된 h5 파일로 평가만 할 때 사용
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


### load_VGG16 클래스 ###
class Load_Trained_VGG16:
    def __init__(self, path, retrain=False):
        self.model = None
        self.result = None
        self.path = path
        self.retrain = retrain
        self.Load_model()

        if generator_train and generator_test:
            generator_train.reset()
            generator_test.reset()

        if retrain:
            self.Retrain_model(1)  # (1)은 에폭수의 예시임
            self.Model_save()
            generator_train.reset()
            generator_test.reset()

        # self.Evaluate_model()

        else:
            self.Load_model()
            # self.Evaluate_model()
            generator_train.reset()
            generator_test.reset()

    def Load_model(self):
        self.model = load_model(self.path)
        return self.model

    def Model_save(self):

        self.path = self.path[:-1] + 'retrain.h5'
        self.model.save(self.path)


        # def Evaluate_model(self):
        # generator_test.reset()
        # self.result = self.model.evaluate_generator(generator_test)


        # for name, value in zip(self.model.metrics_names, self.result):
        #  print(name, value)

    def Retrain_model(self, epoch):
        self.epochs = epoch

        self.model.fit_generator(generator=generator_train,
                                 epochs=self.epochs,
                                 steps_per_epoch=steps_per_epoch_train,
                                 class_weight=class_weight,
                                 validation_data=generator_test,
                                 validation_steps=steps_per_epoch_val)

    ### 분류 보고서 작성 객체 함수 ###
    def Classification_report(self):
        ## cls_pred = np.argmax(y_pred, axis=1)
        ## y_pred = self.model.predict_generator(generator_test, steps=steps_test)
        ## steps_test = generator_test.n / batch_size
        # cm = confusion_matrix(y_true = cls_test, y_pred=cls_pred)
        # re = classification_report(y_true = cls_test, y_pred=cls_pred)
        ## cls_test = generator_test.classes

        generator_test.reset()
        cls_test = generator_test.classes
        steps_test = generator_test.n / batch_size
        y_pred = self.model.predict_generator(generator_test, steps=steps_test)
        cls_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)
        re = classification_report(y_true=cls_test, y_pred=cls_pred)
        print(cm)
        print(re)


        ### 메인 ####


if __name__ == "__main__":
    ##### 사진 로드 ########

    datagen_train = ImageDataGenerator(rescale=1. / 128)
    datagen_test = ImageDataGenerator(rescale=1. / 128)
    batch_size = 10
    steps_per_epoch_train = 402 // batch_size  # 원본 훈련용 사진 402 대신 training set의 개수를 넣어주셔도 됩니다.
    steps_per_epoch_val = 201 // batch_size  # 원본 검증용 사진 201 대신 test set의 개수를 넣어주셔도 됩니다.

    #### train_dir, test_dir 에 훈련용, 검증용 사진 경로를 지정해 주세요
    # train_dir = './colabData/Colab Notebooks/dicern/train1'
    # test_dir = './colabData/Colab Notebooks/dicern/valid1'
    train_dir = '/home/turbo/share/sda1/data/train5'
    # test_dir = '/home/turbo/share/sda1/data/valid10'
    # test_dir = '/home/turbo/share/sda1/data/origin_divide/valid'
    test_dir = '/home/turbo/share/sda1/data/simple_divide/valid'

    #### directory에 훈련용 사진 경로를 넣어 주세요
    generator_train = datagen_train.flow_from_directory(directory=train_dir,

                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        class_mode='categorical', shuffle=True)
    #### directory에 테스트용 사진 경로를 넣어 주세요
    generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                      target_size=(224, 224),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    #### class imbalance 계산
    cls_train = generator_train.classes
    cls_test = generator_test.classes

    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(cls_train), y=cls_train)

    #### 클래스 실행

    #### 경로지정
    # path = './colabData/Colab Notebooks/dicern/vgg16_best.1'
    # path = '/home/turbo/study/playground/pet_resnet50/model/vgg16_e7.h5'
    path = '/home/turbo/study/playground/pet_resnet50/model/vgg16_e12.h5'
    test = Load_Trained_VGG16(path)
    #### test = Load_Trained_VGG16(path, retrain=True) , 즉 retrain=True를 함수 인자로 넣어주면 기존 모델을 재훈련 시킴
    ##### Classification_report()를 실행하면 평가를 해 줌
    test.Classification_report()