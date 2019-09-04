import tensorflow as tf
import layers.tf_utils as tf_utils


def diag_offdiag_maxpool(input):
    """
    Takes the max values over the diagonal (vertices) and offdiagonal (edges)
    :param input: Tensor of shape BxSxNxN
    :return: Tensor of shape Bx2S
    """
    max_diag = tf.reduce_max(tf.matrix_diag_part(input), axis=2)  # BxS

    max_val = tf.reduce_max(max_diag)

    min_val = tf.reduce_max(tf.multiply(input, tf.constant(-1.)))
    val = tf.abs(max_val+min_val)
    min_mat = tf.expand_dims(
        tf.expand_dims(
            tf.matrix_diag(
                tf.add(
                    tf.multiply(
                        tf.matrix_diag_part(input[0][0]),
                        0),
                    val)
            ),
            axis=0),
        axis=0)
    max_offdiag = tf.reduce_max(tf.subtract(input, min_mat), axis=[2, 3])

    return tf.concat([max_diag, max_offdiag], axis=1)  # output Bx2S


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    activation_fn=tf.nn.relu):
    """
    Fully connected layer with non-linear operation.
    :param inputs: 2D tensor of size B x N
    :param num_outputs: int
    :param scope: scope for operation
    :param activation_fn: function to apply after layer
    :return: Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        initializer = tf.contrib.layers.xavier_initializer()
        weights = tf.get_variable("weights", shape=[num_input_units, num_outputs],
                                  initializer=initializer, dtype=tf.float32)

        outputs = tf.matmul(inputs, weights)
        biases = tf.get_variable('biases', [num_outputs],
                                  initializer=tf.constant_initializer(0.))

        outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def skip_connection(in1, in2, output_depth, name, is_training=None):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param output_depth: output num of features
    :param name: name for the scope
    :param is_training: is_training
    :return: Tensor of shape N x output_depth x m x m
    """
    assert is_training is not None, 'is_training flag is needed for skip_connection with concatenation'

    out = tf.concat([in1, in2], axis=1)

    # reduce dimension back to d2
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    out = tf_utils.conv2d(out, output_depth, [1, 1], padding='VALID', stride=[1, 1], is_training=is_training, scope=name)
    out = tf.transpose(out, perm=[0, 3, 1, 2])

    return out
