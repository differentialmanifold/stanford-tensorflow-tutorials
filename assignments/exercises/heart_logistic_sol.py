import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset

N_FEATURES = 9
BATCH_SIZE = 5
N_CLASS = 2

learning_rate = 0.0001
n_epochs = 30
print_size = 100

X = tf.placeholder(tf.float32, [None, N_FEATURES], name='X_placeholder')
Y = tf.placeholder(tf.int32, [None, N_CLASS], name='Y_placeholder')

w = tf.Variable(tf.random_normal(shape=[N_FEATURES, N_CLASS], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, N_CLASS]), name="bias")

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

filename = 'C:/Users/Yi/workspace/project/tensorflow-collection/stanford-tensorflow-tutorials/examples/data/heart.csv'

dataset = TextLineDataset(filename).skip(1)

record_defaults = [[1.0] for _ in range(N_FEATURES)]
record_defaults[4] = ['']
record_defaults.append([1])


def parse_csv(line_str):
    content = tf.decode_csv(line_str, record_defaults)
    content[4] = tf.cond(tf.equal(content[4], tf.constant('Present')), lambda: tf.constant(1.0),
                         lambda: tf.constant(0.0))
    features = tf.stack(content[:N_FEATURES])
    label = tf.one_hot(content[-1], 2)
    return features, label


dataset = dataset.map(parse_csv)

train_dataset = dataset.take(412).shuffle(100).batch(BATCH_SIZE).repeat()

test_dataset = dataset.skip(412).batch(BATCH_SIZE)

train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

train_next_element = train_iterator.get_next()
test_next_element = test_iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_num = 412
    n_batches = int(train_num / BATCH_SIZE)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = sess.run(train_next_element)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Optimization Finished!')

    test_num = 50
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(test_num / BATCH_SIZE)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = sess.run(test_next_element)
        accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds / test_num))
