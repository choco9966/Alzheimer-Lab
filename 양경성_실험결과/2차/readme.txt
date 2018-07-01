vgg16
시트1 : 30000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
2018-06-25
기존 자료 뻥튀기 30,000개 4:1 분리 5번 실행
240/240 [==============================] - 285s 1s/step - loss: 0.0418 - categorical_accuracy: 0.9878 - val_loss: 0.0430 - val_categorical_accuracy: 0.9862

Confusion matrix:
[[1980   20    0]
 [   3 1997    0]
 [   1   59 1940]]
             precision    recall  f1-score   support

          0       1.00      0.99      0.99      2000
          1       0.96      1.00      0.98      2000
          2       1.00      0.97      0.98      2000

avg / total       0.99      0.99      0.99      6000



시트 2: 노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
120/120 [==============================] - 91s 756ms/step - loss: 0.0312 - categorical_accuracy: 0.9920 - val_loss: 0.0561 - val_categorical_accuracy: 0.9793

Confusion matrix:
[[996   2   2]
 [  7 944  49]
 [  2   0 998]]
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      1000
          1       1.00      0.94      0.97      1000
          2       0.95      1.00      0.97      1000

avg / total       0.98      0.98      0.98      3000



시트 3: 단순화시킨 이미지(등고선 스타일)로 노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음

"120/120 [==============================] - 88s 732ms/step - loss: 0.0051 - categorical_accuracy: 0.9996 - val_loss: 0.0110 - val_categorical_accuracy: 0.9970

Confusion matrix:
[[998   1   1]
 [  3 994   3]
 [  0   1 999]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      1000
          1       1.00      0.99      1.00      1000
          2       1.00      1.00      1.00      1000

avg / total       1.00      1.00      1.00      3000

