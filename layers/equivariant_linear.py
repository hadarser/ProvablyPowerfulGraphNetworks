import tensorflow as tf
import layers.tf_utils as tf_utils


def equi2_to_2_only_mlp(name, inputs, is_training, bn_decay, num_features, depth_of_mlp):
    """
    MLP layer(s)
    :param name: scope name
    :param inputs: input tensor shape BxSxNxN
    :param is_training: flag
    :param bn_decay:
    :param num_features: num of output features
    :param depth_of_mlp: num of sequential mlp layers
    :return: tensor shape of Bx num_features xNxN
    """
    output = tf.transpose(inputs, perm=[0, 2, 3, 1])

    for i in range(1, depth_of_mlp+1):
        output = tf_utils.conv2d(output, num_features, [1, 1], padding='VALID', stride=[1, 1], bn=False,
                                 is_training=is_training, scope=name+'_conv{}'.format(i), bn_decay=bn_decay)

    output = tf.transpose(output, perm=[0, 3, 1, 2])
    return output
