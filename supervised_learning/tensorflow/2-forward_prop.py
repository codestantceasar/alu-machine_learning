#!/usr/bin/env python3
"""Forward propagation"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates forward propagation graph

    x: input placeholder
    layer_sizes: nodes per layer
    activations: activations per layer

    Returns:
        prediction tensor
    """
    output = x

    for i in range(len(layer_sizes)):
        output = create_layer(
            output,
            layer_sizes[i],
            activations[i]
        )

    return output
