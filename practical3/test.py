import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 50
NUM_BATCHES = 3000
TEST_ID = 100
TEST_SIZE = 2000
TOP_PATCHES = 12
VISUAL_FILTERS = 5


def get_tests(num_tests: int, is_valid: bool=True):
    images = mnist.validation.images if is_valid else mnist.test.images
    labels = mnist.validation.labels if is_valid else mnist.test.labels
    x_valid = np.reshape(images[:num_tests, :], (num_tests, 28, 28, 1))
    y_valid = labels[:num_tests, :]
    return x_valid, y_valid


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2dl1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def conv2dl2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def visualize_layer(W):
    fig = plt.figure()
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(W[:, :, :, i].reshape(12, 12), cmap='Greys_r')
    fig.show()


def visualize_patches(H, x_test):
    sort_a = np.argsort(H, axis=None)
    idx = np.unravel_index(sort_a, H.shape)
    test_idx = np.zeros(TOP_PATCHES, dtype=np.int)
    filter_idx = np.zeros(TOP_PATCHES, dtype=np.int)
    activation_set = set()
    cur_idx = 0
    '''
    for i in range(idx[0].shape[0]):
        tuple_act = (idx[0][i], idx[3][i])
        if tuple_act not in activation_set:
            test_idx[cur_idx] = tuple_act[0]
            filter_idx[cur_idx] = tuple_act[1]
            cur_idx += 1
            activation_set.add(tuple_act)
            if cur_idx >= TOP_PATCHES:
                break
    '''
    test_idx = idx[0][:TOP_PATCHES]
    width_idx =idx[1][:TOP_PATCHES] * 2
    height_idx = idx[2][:TOP_PATCHES] * 2
    '''
    patches = x_test[test_idx, width_idx: width_idx + np.repeat(12, TOP_PATCHES),
                    height_idx: height_idx + np.repeat(12, TOP_PATCHES), :]
'''

    fig = plt.figure()
    for i in range(TOP_PATCHES):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        m = x_test[test_idx[i], width_idx[i]: width_idx[i] + 12,
                    height_idx[i]: height_idx[i] + 12, :]
        ax.imshow(np.reshape(m, [12, 12]), cmap='Greys_r')
    fig.show()


if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Set up placeholders
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Make the fully connected layer
    # (28 - 12) / 2  + 1 = 9 is a size of matrix.
    W_conv1 = weight_variable([12, 12, 1, 25])
    b_conv1 = bias_variable([25])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2dl1(x_image, W_conv1) + b_conv1)

    # Make second fully connected layer
    W_conv2 = weight_variable([5, 5, 25, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2dl2(h_conv1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)
    # 9 / 2 but same -> 10 / 2 = 5, pool operation reduces dimension.
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

    x_valid, y_valid = get_tests(TEST_SIZE)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(NUM_BATCHES):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = np.reshape(batch_x, (batch_x.shape[0], 28, 28, 1))
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            if i % TEST_ID == TEST_ID - 1:
                print("Finished processing ", i + 1, " batches.")
                train_accs[i // TEST_ID] = sess.run(accuracy, feed_dict={x: batch_x,
                                                                         y_: batch_y, keep_prob: 1.0})
                valid_accs[i // TEST_ID] = sess.run(accuracy, feed_dict={x: x_valid,
                                                                         y_: y_valid, keep_prob: 1.0})
                print('Step {}, training accuracy {:.3f}'.format(i + 1, train_accs[i // TEST_ID]))
                print('Step {}, validation accuracy {:.3f}'.format(i + 1, valid_accs[i // TEST_ID]))
        x_test, y_test = get_tests(10000, False)
        print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
        W = W_conv1.eval(sess)
        visualize_layer(W)
        H = sess.run(h_conv1, feed_dict={x: x_test})[:, :, :, :VISUAL_FILTERS]
        visualize_patches(H, x_test)
        t=1
