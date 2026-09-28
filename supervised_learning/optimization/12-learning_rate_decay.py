#!/usr/bin/env python3
"""Learning rate decay operation"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """Creates decay operation"""

    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )
