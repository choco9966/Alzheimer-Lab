시트1 : 기존자료 603개를 그대로 돌린 결과
2018-06-24
기존 자료 603개  3:1 분리 300번 실행
1100/1100 [==============================] - 313s 285ms/step - loss: 0.6308 - categorical_accuracy: 0.7667 - val_loss: 0.7048 - val_categorical_accuracy: 0.6714

Confusion matrix: 
[[73  0  0]  
[19  0  1]  
[30  0 29]]              
precision    recall  f1-score   support           
0       0.60      1.00      0.75        73           
1       0.00      0.00      0.00        20           
2       0.97      0.49      0.65        59 

avg / total       0.66      0.67      0.61       152

시트2 : 30000개로 뻥튀기 한 후 5번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음
2018-06-25
기존 자료 뻥튀기 30,000개 4:1 분리 5번 실행
202s 168ms/step - loss: 1.0641 - categorical_accuracy: 0.4440 - val_loss: 0.9442 - val_categorical_accuracy: 0.6017

Confusion matrix:
[[1115  577  308]
 [ 384  818  798]
 [  90  233 1677]]
             precision    recall  f1-score   support

          0       0.70      0.56      0.62      2000
          1       0.50      0.41      0.45      2000
          2       0.60      0.84      0.70      2000

avg / total       0.60      0.60      0.59      6000

시트 3: 30000개로 뻥튀기 한 후 200번 돌린 결과
         뻥튀기 했으므로  steps_per_epoch=steps_per_epoch_train*50 은 하지 않음

2018-06-26
기존 자료 뻥튀기 30,000개 4:1 분리 200번 실행
1200/1200 [==============================] - 201s 168ms/step - loss: 0.9311 - categorical_accuracy: 0.5537 - val_loss: 0.8267 - val_categorical_accuracy: 0.629


Confusion matrix:
[[ 906  908  186]
 [ 162 1244  594]
 [  16  356 1628]]
             precision    recall  f1-score   support

          0       0.84      0.45      0.59      2000
          1       0.50      0.62      0.55      2000
          2       0.68      0.81      0.74      2000

avg / total       0.67      0.63      0.63      6000


1 을 잘 구별 못하는 듯.(1을 2와 헤깔리는 듯)
그래서 잡음과 조명 효과를 빼고 5000개씩 총 15000개로 다시 테스트 예정
