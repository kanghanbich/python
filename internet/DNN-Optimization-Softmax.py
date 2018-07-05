import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28], name="x")
Y = tf.placeholder(tf.float32, [None, 10], name="y")

w = tf.Variable(tf.random_normal([28*28, 10]), name="Weight")
b = tf.Variable(tf.random_normal([10]), name="Bias")

H = tf.nn.softmax(tf.matmul(X, w) + b)

with tf.name_scope("Train") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H, Y))
    cost_summ = tf.summary.scalar("Cost", cost)
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    correct = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summ = tf.summary.scalar("Accuracy", accuracy, ["Epoch"])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    epoch_merged = tf.summary.merge_all("Epoch")
    writer = tf.summary.FileWriter("./logs", sess.graph)
    
    batch_size = mnist.train.num_examples/100
    
    for epoch in xrange(15):
        for step in xrange(batch_size):
            image, label = mnist.train.next_batch(100)
            summ, _ = sess.run([merged, train], feed_dict={X: image, Y: label})
            writer.add_summary(summ, epoch * batch_size + step)
        
        summ, _accuracy = sess.run([epoch_merged, accuracy],
                                          feed_dict={X: mnist.test.images,
                                                     Y: mnist.test.labels}) 
        writer.add_summary(summ, (epoch + 1) * batch_size)
        
        print ("Epoch:", (epoch + 1))
        print ("Test Accuracy:", _accuracy)

