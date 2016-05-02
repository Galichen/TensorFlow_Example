#-*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

# Numpy 랜덤으로 100개의 가짜 데이터 채우기. (float64 -> float32로 변환)
x_data = np.float32(np.random.rand(2, 100))
# 학습 레이블(목표값)은 아래의 식으로 산출. (W = [0.1, 0.2], b = 0.3)
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 선형모델 정의
# y = [w vector][x vector] + b
# bias value는 0으로 초기화
b = tf.Variable(tf.zeros([1]))
# W는 1x2 형태의 웨이트 변수 (균등 랜덤값으로 초기화)
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 평균 제곱 오차(MSE(Mean Square Error))를 에러 함수로 지정
loss = tf.reduce_mean(tf.square(y - y_data))
# 경사하강법으로 MSE를 최소화하는 학습 오퍼레이션 적용 (0.5는 학습 비율)
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


##################
## 학습 세션 시작 ##
##################

# 모든 변수를 초기화.
init = tf.initialize_all_variables()

# 세션 시작
sess = tf.Session()
sess.run(init)

# 1000번 학습.
for step in xrange(0, 1001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)