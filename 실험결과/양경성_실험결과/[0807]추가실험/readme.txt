vgg16

실험 목표
1. 노이즈.빛 없이		
2. 데이터 수 줄여보기
3. 테스트 데이타는 원본 사용

----------------------
1. 노이즈.빛 없이
----------------------
노이즈, 빛의 효과는 미비한 듯, 보통 데이타에 노이즈, 빛을 이용해 뻥튀기하면 정확도가 조금 떨어짐.
단순화한 데이타는 차이가 없음.(단순화 과정에서 노이즈,빛의 효과 상쇄되서 그러듯)

시트2 : 4:1로 분리한 후  노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 100번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
120/120 [==============================] - 80s 669ms/step - loss: 0.0280 - categorical_accuracy: 0.9938 - val_loss: 0.1241 - val_categorical_accuracy: 0.9487

Confusion matrix:
[[996   4   0]
 [  7 863 130]
 [  0  13 987]]
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      1000
          1       0.98      0.86      0.92      1000
          2       0.88      0.99      0.93      1000

avg / total       0.95      0.95      0.95      3000


==> 4:1로 분리한 후  노이즈, 조명효과 있는 15000개로 뻥튀기 한 후 100번 돌린 결과
120/120 [==============================] - 91s 756ms/step - loss: 0.0667 - categorical_accuracy: 0.9854 - val_loss: 0.1847 - val_categorical_accuracy: 0.9263

Confusion matrix:
[[996   4   0]
 [ 38 791 171]
 [  0   8 992]]
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      1000
          1       0.99      0.79      0.88      1000
          2       0.85      0.99      0.92      1000

avg / total       0.93      0.93      0.92      3000


시트 3: 단순화시킨 이미지(등고선 스타일)로  4:1로 분리한 후 노이즈, 조명효과 뺀 15000개로 뻥튀기 한 후 62번 돌린 결과
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

====>
 단순화시킨 이미지(등고선 스타일)로  4:1로 분리한 후 노이즈, 조명효과 있는 15000개로 뻥튀기 한 후 62번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
120/120 [==============================] - 91s 762ms/step - loss: 0.0267 - categorical_accuracy: 0.9964 - val_loss: 0.1028 - val_categorical_accuracy: 0.968

[[986  13   1]
 [ 19 923  58]
 [  0   3 997]]
             precision    recall  f1-score   support

          0       0.98      0.99      0.98      1000
          1       0.98      0.92      0.95      1000
          2       0.94      1.00      0.97      1000

avg / total       0.97      0.97      0.97      3000

----------------------
2. 데이터 수 줄여보기
----------------------
보통 데이터와 단순화시킨 데이타 모두에 대해서는 15,000개와 비교해보면 3,000개부터 거의 유사한 결과가 나옴

시트 4: 기존 자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 15,000개 100번 실행
20/120 [==============================] - 80s 666ms/step - loss: 0.0279 - categorical_accuracy: 0.9937 - val_loss: 0.0883 - val_categorical_accuracy: 0.9607

Confusion matrix:
[[979  21   0]
 [  2 986  12]
 [  0  29 971]]
             precision    recall  f1-score   support

          0       1.00      0.98      0.99      1000
          1       0.95      0.99      0.97      1000
          2       0.99      0.97      0.98      1000

avg / total       0.98      0.98      0.98      3000

====> 기존 자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 1500개 100번 실행
12/12 [==============================] - 8s 639ms/step - loss: 0.2088 - categorical_accuracy: 0.9700 - val_loss: 0.2918 - val_categorical_accuracy: 0.9300

Confusion matrix:
[[100   0   0]
 [  5  87   8]
 [  1   5  94]]
             precision    recall  f1-score   support

          0       0.94      1.00      0.97       100
          1       0.95      0.87      0.91       100
          2       0.92      0.94      0.93       100

avg / total       0.94      0.94      0.94       300

======> 기존 자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 3000개 100번 실행
24/24 [==============================] - 15s 629ms/step - loss: 0.1339 - categorical_accuracy: 0.9746 - val_loss: 0.1870 - val_categorical_accuracy: 0.9650

Confusion matrix:
[[196   3   1]
 [  2 188  10]
 [  0   4 196]]
             precision    recall  f1-score   support

          0       0.99      0.98      0.98       200
          1       0.96      0.94      0.95       200
          2       0.95      0.98      0.96       200

avg / total       0.97      0.97      0.97       600


=====> 기존 자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 4500개 100번 실행
36/36 [==============================] - 23s 639ms/step - loss: 0.0808 - categorical_accuracy: 0.9900 - val_loss: 0.1180 - val_categorical_accuracy: 0.9767

Confusion matrix:
[[294   6   0]
 [  2 291   7]
 [  0   8 292]]
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       300
          1       0.95      0.97      0.96       300
          2       0.98      0.97      0.97       300

avg / total       0.97      0.97      0.97       900


========> 기존 자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 6000개 100번 실행
48/48 [==============================] - 30s 634ms/step - loss: 0.0594 - categorical_accuracy: 0.9931 - val_loss: 0.1289 - val_categorical_accuracy: 0.9458

[[397   3   0]
 [  4 373  23]
 [  0  10 390]]
             precision    recall  f1-score   support

          0       0.99      0.99      0.99       400
          1       0.97      0.93      0.95       400
          2       0.94      0.97      0.96       400

avg / total       0.97      0.97      0.97      1200

시트 5: 단순화한  자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 15,000개  62번 실행
120/120 [==============================] - 82s 680ms/step - loss: 0.0166 - categorical_accuracy: 0.9975 - val_loss: 0.0818 - val_categorical_accuracy: 0.9733

Confusion matrix:
[[990  10   0]
 [  7 929  64]
 [  0   2 998]]
             precision    recall  f1-score   support

          0       0.99      0.99      0.99      1000
          1       0.99      0.93      0.96      1000
          2       0.94      1.00      0.97      1000

avg / total       0.97      0.97      0.97      3000

========> 단순화한  자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 1500개 100번 실행
12/12 [==============================] - 8s 659ms/step - loss: 0.1043 - categorical_accuracy: 0.9900 - val_loss: 0.1971 - val_categorical_accuracy: 0.9600

Confusion matrix:
[[88 12  0]
 [ 1 97  2]
 [ 0  2 98]]
             precision    recall  f1-score   support

          0       0.99      0.88      0.93       100
          1       0.87      0.97      0.92       100
          2       0.98      0.98      0.98       100

avg / total       0.95      0.94      0.94       300


======> 단순화한  자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 3000개 100번 실행
24/24 [==============================] - 16s 649ms/step - loss: 0.0483 - categorical_accuracy: 0.9958 - val_loss: 0.1268 - val_categorical_accuracy: 0.9733

Confusion matrix:
[[195   5   0]
 [  2 191   7]
 [  0   2 198]]
             precision    recall  f1-score   support

          0       0.99      0.97      0.98       200
          1       0.96      0.95      0.96       200
          2       0.97      0.99      0.98       200

avg / total       0.97      0.97      0.97       600

========> 단순화한  자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 4500개 100번 실행
36/36 [==============================] - 24s 656ms/step - loss: 0.0322 - categorical_accuracy: 0.9967 - val_loss: 0.1028 - val_categorical_accuracy: 0.9622

Confusion matrix:
[[290  10   0]
 [  1 277  22]
 [  0   1 299]]
             precision    recall  f1-score   support

          0       1.00      0.97      0.98       300
          1       0.96      0.92      0.94       300
          2       0.93      1.00      0.96       300

avg / total       0.96      0.96      0.96       900


=========> 단순화한  자료 4:1 분리 후 뻥튀기(노이즈,밝기 없음) 6000개 100번 실행
48/48 [==============================] - 31s 655ms/step - loss: 0.0225 - categorical_accuracy: 0.9971 - val_loss: 0.0789 - val_categorical_accuracy: 0.9717

Confusion matrix:
[[394   6   0]
 [  2 374  24]
 [  0   2 398]]
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       400
          1       0.98      0.94      0.96       400
          2       0.94      0.99      0.97       400

avg / total       0.97      0.97      0.97      1200

----------------------------
3. 테스트 데이타는 원본 사용
----------------------------
뻥튀기된 valid로 나뉜 부분으로 테스트한 것보다 원데이타 중 valid로 나뉜 부분 119에 대해 평가한 것이 더 정확도가 높았음
단, 단순화된 데이타에서는 테스트도 남겨둔 119개를 단순화시킨 후 테스트하였음.

시트 2,4: 하단 노란 박스가 119로 테스트한 부분임

저장된 모델 로드하여 원데이타 중 valid로 나뉜 부분 119에 대해 평가
[[58  0  0]
 [ 0 14  1]
 [ 0  1 45]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        58
          1       0.93      0.93      0.93        15
          2       0.98      0.98      0.98        46

avg / total       0.98      0.98      0.98       119



시트 3,5: 하단 노란 박스가 119로 테스트한 부분임
저장된 모델 로드하여 단순화한 데이타 중 valid로 나뉜 부분 119에 대해 평가

[[58  0  0]
 [ 0 14  1]
 [ 0  0 46]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        58
          1       1.00      0.93      0.97        15
          2       0.98      1.00      0.99        46

avg / total       0.99      0.99      0.99       119

