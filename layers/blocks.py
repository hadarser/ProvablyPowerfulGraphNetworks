import tensorflow as tf
import layers.equivariant_linear as eq
import layers.layers as layers


def regular_block(inputs, name, config, output_depth, is_training):
    # Inputs : N x input_depth x m x m
    block = inputs

    # MLP route 1
    mlp1 = eq.equi2_to_2_only_mlp(name + '_mlp1', block, is_training, bn_decay=None, num_features=output_depth,
                                  depth_of_mlp=config.architecture.depth_of_mlp)

    # MLP route 2
    mlp2 = eq.equi2_to_2_only_mlp(name + '_mlp2', block, is_training, bn_decay=None, num_features=output_depth,
                                  depth_of_mlp=config.architecture.depth_of_mlp)
    block = tf.matmul(mlp1, mlp2)

    # Skip-connection with reducing dimension back to output depth
    block = layers.skip_connection(in1=inputs, in2=block, output_depth=output_depth, name=name, is_training=is_training)

    return block
