
# coding: utf-8

# In[3]:


#MNIST 데이터셋 불러오
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#TensorFlow InteractiveSession start
import tensorflow as tf
sess = tf.InteractiveSession()

#입력될 이미지:x 한줄로28*28, 출력 클래스:_y
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#가중치:W 편향:b, 모델 매개변수
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#바이블(variable)초기화
sess.run(tf.initialize_all_variables())

#입력 이미지 * 가중치 + 편향(임의의 값)그냥더하는거
y = tf.nn.softmax(tf.matmul(x,W) + b)
#tf.reduce_sum은 모든 클래스에 대하여 결과를 합하는 함수
#tf.reduce_mean은 사용된 이미지들 각각에서 계산된 합의 평균을 구하는 함수
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#학습속도 0.5의 경사 하강법
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#50개의 훈련 샘플
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#모델 평가
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#가중치 초기
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#padding & stride를 입력, 스트라이드는 1, 패딩은 출력과 입력의
#크기가 같게 되도록 0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#W_conv1 = weight_variable([5, 5, 1, 32]), 첫번째 합성곱 계층
#5*5 크기, 32개의 필터, 입력 채널의 수, 출력 채널의 수
W_conv1 =tf.get_variable("W1", shape=[5, 5, 1, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = bias_variable([32])
#
x_image = tf.reshape(x, [-1,28,28,1])
#
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_pool1 = max_pool_2x2(h_conv1)

#W_conv2 = weight_variable([5, 5, 32, 64]), 두번째 합성곱 계층
#5*5 크기, 64개의 필터, 입력 채널의 수, 출력 채널의 수
W_conv2=tf.get_variable("W2", shape=[5, 5, 32, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
b_conv2 = bias_variable([64])
#
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_pool2 = max_pool_2x2(h_conv2)

#Fully-Connected Layer, 두번째 계층을 거친 뒤 크기 7*7
#1024개의 뉴런으로 연결되는 완전 연결 계층
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

