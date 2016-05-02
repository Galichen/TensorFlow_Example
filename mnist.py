#-*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import input_data

# 트레이닝 데이터 셋
# 반환값은 55000개의 28*28의 픽셀값의 벡터, 즉 784-dimension 벡터
# 결과 레이블은 0~9사이의 값이며 one-hot-encoding이 필요
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 트레이닝 데이터 셋용 플레이스 홀더
x = tf.placeholder("float",[None,784])

# 784-dimension 이미지 벡터를 x에 곱해 10-dimension의 결과를 내기위해 W를 [784,10]의 행렬로 정의
# b는 결과에 더해줄 bias 상수이므로 10-dimension vector
# 모두 0으로 초기화
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 선형 모델 구현
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 정답 레이블용 플레이스 홀더
y_ = tf.placeholder("float", [None, 10])

# Error 함수 정의
# 정보이론의 크로스 엔트로피 공식을 써서 정답과 결과의 엔트로피가 최소가 되는 방향으로 훈련시킨다.
cross_entrophy = -tf.reduce_sum(y_*tf.log(y));

# 학습률 에타 정의
etha = 0.01

#학습 오퍼레이션 정의
train_step = tf.train.GradientDescentOptimizer(etha).minimize(cross_entrophy);

# 모든 변수 초기화
init = tf.initialize_all_variables();
sess = tf.Session()
sess.run(init)

#100개씩 미니배치로 1000번 Stochastic Training

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 트레이닝 모델 테스팅 : mnist.test.image의 데이터 활용
# 테스트 셋을 활용하여 모델의 정확도 측정

# 정답과 모델의 아웃풋이 일치하면 correct
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1));

# 정확도 평균 계산
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
print "--Accuracy--"
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



