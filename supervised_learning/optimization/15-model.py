#!/usr/bin/env python3
"""Model"""

import tensorflow as tf
import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('14-batch_norm').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """builds, trains and saves a neural network"""

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x, y = create_placeholders(nx, classes)

    y_pred = forward_prop(x, layers, activations)

    loss = calculate_loss(y, y_pred)

    accuracy = calculate_accuracy(y, y_pred)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)

    alpha_decay = learning_rate_decay(
        alpha,
        decay_rate,
        global_step,
        1
    )

    # REMOVED: global_step=global_step here to prevent per-mini-batch updates
    train_op = tf.train.AdamOptimizer(
        learning_rate=alpha_decay,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    m = Data_train[0].shape[0]
    steps = (m + batch_size - 1) // batch_size

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):

            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={
                    x: Data_train[0],
                    y: Data_train[1]
                }
            )

            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={
                    x: Data_valid[0],
                    y: Data_valid[1]
                }
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            X_shuff, Y_shuff = shuffle_data(
                Data_train[0],
                Data_train[1]
            )

            for step in range(steps):

                start = step * batch_size
                end = min(start + batch_size, m)

                X_batch = X_shuff[start:end]
                Y_batch = Y_shuff[start:end]

                sess.run(
                    train_op,
                    feed_dict={
                        x: X_batch,
                        y: Y_batch
                    }
                )

                if (step + 1) % 100 == 0:
                    cost, acc = sess.run(
                        [loss, accuracy],
                        feed_dict={
                            x: X_batch,
                            y: Y_batch
                        }
                    )

                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(cost))
                    print("\t\tAccuracy: {}".format(acc))

            # ADDED: Increment global_step manually once per epoch
            sess.run(global_step.assign_add(1))

        return saver.save(sess, save_path)
