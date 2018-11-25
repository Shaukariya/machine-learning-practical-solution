import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

BATCH_SIZE = 50
TEST_ID = 50


def reshape_data(X, y):
    N = X.shape[0]
    X = np.reshape(X, (N, 28, 28, 1))
    y_oh = np.zeros((N, 10))
    y_oh[np.arange(N), y] = 1
    return X, y_oh


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2dl1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


def conv2dl2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    '''
    fig = plt.figure()
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(x_train[i].reshape(28, 28), cmap='Greys_r')
    fig.show()
'''
    # Set up placeholders
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Make the fully connected layer
    W_conv1 = weight_variable([12, 12, 1, 25])
    b_conv1 = bias_variable([25])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2dl1(x_image, W_conv1) + b_conv1)

    # Make second fully connected layer
    W_conv2 = weight_variable([5, 5, 25, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2dl2(h_conv1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])

    W_fc1 = weight_variable([5 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_accs = np.zeros(100)
    valid_accs = np.zeros(100)

    x_train, y_train = reshape_data(x_train, y_train)
    x_test, y_test = reshape_data(x_test, y_test)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        N = x_train.shape[0]
        for i in range(N // BATCH_SIZE):
            batch_x = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
            batch_y = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            if i % TEST_ID == 0:
                print("Finished processing ", (i // 10) + 1, " batches.")

                train_accs[i // TEST_ID] = sess.run(accuracy, feed_dict={x: batch_x,
                                                                    y_: batch_y, keep_prob: 1.0})

                valid_accs[i // TEST_ID] = sess.run(accuracy, feed_dict={x: x_test[:1000],
                                                                    y_: y_test[:1000], keep_prob: 1.0})
                print('Step {}, training accuracy {:.3f}'.format(i + 1, train_accs[i // TEST_ID]))
                print('Step {}, validation accuracy {:.3f}'.format(i + 1, valid_accs[i // TEST_ID]))
        sess.close()
    print(train_accs)
    print(valid_accs)
