vgg16
시트1 : 4:1로 분리한 후 30000개로 뻥튀기 한 후 90번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
2018-07-04
기존 자료 뻥튀기 30,000개 4:1 분리 5번 실행
240/240 [==============================] - 173s 722ms/step - loss: 0.0273 - categorical_accuracy: 0.9943 - val_loss: 0.1289 - val_categorical_accuracy: 0.9503

Confusion matrix:
[[1937   60    3]
 [  23 1793  184]
 [   0   28 1972]]
             precision    recall  f1-score   support

          0       0.99      0.97      0.98      2000
          1       0.95      0.90      0.92      2000
          2       0.91      0.99      0.95      2000

avg / total       0.95      0.95      0.95      6000


시트 2: 4:1로 분리한 후  노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
20/120 [==============================] - 80s 666ms/step - loss: 0.0279 - categorical_accuracy: 0.9937 - val_loss: 0.0883 - val_categorical_accuracy: 0.9607

Confusion matrix:
[[997   3   0]
 [ 12 906  82]
 [  1  20 979]]
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      1000
          1       0.98      0.91      0.94      1000
          2       0.92      0.98      0.95      1000

avg / total       0.96      0.96      0.96      3000


시트 3: 단순화시킨 이미지(등고선 스타일)로  4:1로 분리한 후 노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음

120/120 [==============================] - 82s 680ms/step - loss: 0.0166 - categorical_accuracy: 0.9975 - val_loss: 0.0818 - val_categorical_accuracy: 0.9733

Confusion matrix:
[[994   6   0]
 [  8 928  64]
 [  0   2 998]]
             precision    recall  f1-score   support

          0       0.99      0.99      0.99      1000
          1       0.99      0.93      0.96      1000
          2       0.94      1.00      0.97      1000

avg / total       0.97      0.97      0.97      3000


