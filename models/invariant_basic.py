from models.base_model import BaseModel
import layers.equivariant_linear as eq
import layers.layers as layers
import layers.blocks as blocks
import tensorflow as tf


class invariant_basic(BaseModel):
    def __init__(self, config, data):
        super(invariant_basic, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        Build the model computation graph.
        It is made of 2 steps -
            1) the computation layers, similar for benchmark graphs and QM9 graphs,
            2) loss and optimizer (different for QM9).
        :return:
        """
        if self.config.dataset_name != 'QM9':
            # Benchmark data sets - COLLAB, NCI1, NCI109 etc.
            self.labels = tf.placeholder(tf.int32, shape=[None])  # y data
            self.add_loss_optimizer_to_benchmark_model(self.build_model_layers())
        else:
            # QM9 dataset
            if self.config.target_param is False:   # (0 == False) while (0 is not False)
                self.labels = tf.placeholder(tf.int32, shape=[None, 12])  # y data
            else:
                self.labels = tf.placeholder(tf.int32, shape=[None, 1])  # y data, only one target
            self.add_loss_optimizer_to_qm9_model(self.build_model_layers())

    def build_model_layers(self):
        # here you build the tensorflow graph of any model you want
        self.is_training = tf.placeholder(tf.bool)

        self.graphs = tf.placeholder(tf.float32, shape=[None, self.config.node_labels + 1, None, None])  # X data

        new_suffix = self.config.architecture.new_suffix  # True or False

        # build network architecture using config file
        net = self.graphs
        net = blocks.regular_block(net, 'b0', self.config, self.config.architecture.block_features[0], self.is_training)
        if new_suffix:
            hidden_outputs = [net]

        for layer in range(1, len(self.config.architecture.block_features)):
            net = blocks.regular_block(net, 'b{}'.format(layer), self.config,
                                       self.config.architecture.block_features[layer], self.is_training)
            if new_suffix:
                hidden_outputs.append(net)

        if not new_suffix:
            # Old suffix implementation - suffix (i) from paper
            net = layers.diag_offdiag_maxpool(net)
            net = layers.fully_connected(net, 512, "fully1")
            net = layers.fully_connected(net, 256, "fully2")
            out = layers.fully_connected(net, self.config.num_classes, "fully3", activation_fn=None)

        # New suffix
        if new_suffix:
            # New suffix implementation - suffix (ii) from paper
            out = 0
            for i, h in enumerate(hidden_outputs):
                pooled_h = layers.diag_offdiag_maxpool(h)
                fully = layers.fully_connected(pooled_h, self.config.num_classes, "fully{}".format(i), activation_fn=None)
                out += fully

        return out

    def add_loss_optimizer_to_benchmark_model(self, partial_model):
        """
        Adds loss and optimizer to the model's end.
        :param partial_model: the partial built model, in the shape of logits (classification)
        :return:
        """
        # define loss function
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=partial_model))
            self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(partial_model, 1, output_type=tf.int32), self.labels), tf.int32))

        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size*20)

        # choose optimizer
        if self.config.hyperparams.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.hyperparams.momentum)
        elif self.config.hyperparams.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def add_loss_optimizer_to_qm9_model(self, partial_model):
        """
        Adds loss and optimizer to the model's end.
        :param partial_model: the partial built model, in the shape of predicted values (regression)
        :return:
        """
        # define loss function
        with tf.name_scope("loss"):
            distances = tf.losses.absolute_difference(labels=self.labels, predictions=partial_model,
                                                      reduction=tf.losses.Reduction.NONE)
            # Returned value of loss is shape of labels - shape: B,12
            self.sum_distances = tf.reduce_sum(distances, axis=0)  # shape: 12,
            self.loss = tf.reduce_sum(self.sum_distances)  # shape: 1,

        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size*20)

        # choose optimizer
        if self.config.hyperparams.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.hyperparams.momentum)
        elif self.config.hyperparams.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.hyperparams.learning_rate,  # Base learning rate.
            global_step*self.config.hyperparams.batch_size,
            decay_step,
            self.config.hyperparams.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-8)
        return learning_rate
