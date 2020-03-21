from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from my_utils import *
from layers import *
from metrics import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 400, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('epsilon', 1.0, "norm length for (virtual) adversarial training ")
flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
flags.DEFINE_float('xi', 1e-4, "small constant for finite difference")
flags.DEFINE_float('alpha', 1.0, "Weight for VAT loss")
flags.DEFINE_bool('mask_vat', False, 'calculate vat loss only on unlabeled data.')
flags.DEFINE_bool('reload', False, 'reload parameter.')
flags.DEFINE_string('model_path', './model/vat/cora/model', 'path to reload model.')
flags.DEFINE_string('model_save_path', './model/vat/cora/model', 'path to save model.')


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# there is a confusing point, the following line will print the default value
# of parameters, but will print the input value after the first call of FLAG
# values
print(FLAGS.flag_values_dict())

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    # support = [preprocess_adj(adj)]
    support = preprocess_adj(adj)
    num_supports = 1
# elif FLAGS.model == 'gcn_cheby':
#     support = chebyshev_polynomials(adj, FLAGS.max_degree)
#     num_supports = 1 + FLAGS.max_degree
#     model_func = GCN
# elif FLAGS.model == 'dense':
#     support = [preprocess_adj(adj)]  # Not used
#     num_supports = 1
#     model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support': tf.sparse_placeholder(tf.float32),
    # 'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# Create model
def logit(x, is_training=True):
    # first layer
    x = tf.nn.dropout(x, 1 - placeholders['dropout'])
    # x = sparse_dropout(x, 1 - placeholders['dropout'],
    #                    placeholders['num_features_nonzero'])
    l1_weights = tf.get_variable(
        'l1_W', shape=[features.shape[1], FLAGS.hidden1],
        initializer=tf.glorot_uniform_initializer()
    )
    # l1_biases = tf.get_variable(
    #     'l1_b', shape=[FLAGS.hidden1], initializer=tf.constant_initializer(0.0)
    # )
    l1_out = tf.sparse_tensor_dense_matmul(
        # placeholders['support'], tf.sparse_tensor_dense_matmul(x, l1_weights)
        placeholders['support'], tf.matmul(x, l1_weights)
    )
    l1_out = tf.nn.relu(l1_out)

    # second layer
    l1_out = tf.nn.dropout(l1_out, 1 - placeholders['dropout'])
    l2_weights = tf.get_variable(
        'l2_W', shape=[FLAGS.hidden1, placeholders['labels'].get_shape().as_list()[1]],
        initializer=tf.glorot_uniform_initializer()
    )
    output = tf.sparse_tensor_dense_matmul(
        placeholders['support'], tf.matmul(l1_out, l2_weights)
    )
    return output


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logits, is_training=True):
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(FLAGS.num_power_iterations):
        # d = FLAGS.xi * get_normalized_vector(d)
        d = FLAGS.xi * tf.nn.l2_normalize(d, axis=1)
        logit_p = logits
        logit_m = logit(x + d, is_training=is_training)
        # dist = kl_divergence_with_logit(logit_p, logit_m)
        if FLAGS.mask_vat:
            dist = my_kld_with_logit_with_mask(logit_p, logit_m,
                                               placeholders['labels_mask'])
        else:
            dist = my_kld_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    # return FLAGS.epsilon * get_normalized_vector(d)
    return FLAGS.epsilon * tf.nn.l2_normalize(d, axis=1)
    # return FLAGS.epsilon * tf.nn.l2_normalize(d, axis=1), d, dist, logit_m


def virtual_adversarial_loss(x, logits, is_training=True, name="vat_loss"):
    r_vadv = generate_virtual_adversarial_perturbation(x, logits, is_training=is_training)
    # r_vadv, r_d, r_dist, r_logit_m = generate_virtual_adversarial_perturbation(
    #     x, logits, is_training=is_training
    # )
    logits = tf.stop_gradient(logits)
    logit_p = logits
    logit_m = logit(x + r_vadv, is_training=is_training)
    if FLAGS.mask_vat:
        vat_loss = my_kld_with_logit_with_mask(logit_p, logit_m,
                                               placeholders['labels_mask'])
    else:
        vat_loss = my_kld_with_logit(logit_p, logit_m)
    return tf.identity(vat_loss, name=name)
    # return tf.identity(vat_loss, name=name), r_vadv, r_d, r_dist, r_logit_m, logit_m


with tf.variable_scope("VGCN") as scope:
    logits = logit(placeholders['features'])
    sup_loss = masked_softmax_cross_entropy(
        logits, placeholders['labels'], placeholders['labels_mask']
    )

    l2_norm = 0.0
    for var in tf.trainable_variables():
        l2_norm += tf.nn.l2_loss(var)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        # vat_loss, r_vadv, r_d, r_dist, r_logit_m, logit_m = \
        #     virtual_adversarial_loss(placeholders['features'], logits)
        vat_loss = virtual_adversarial_loss(placeholders['features'], logits)

        obj_func = sup_loss + FLAGS.weight_decay * l2_norm + \
                   FLAGS.alpha * vat_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    opt_op = optimizer.minimize(obj_func)

    accuracy = masked_accuracy(logits, placeholders['labels'],
                               placeholders['labels_mask'])


# Initialize session
sess = tf.Session()


# Init variables
saver = tf.train.Saver()
if FLAGS.reload:
    saver.restore(sess, FLAGS.model_path)
else:
    sess.run(tf.global_variables_initializer())

# l1_wei = sess.run(tf.trainable_variables()[0])
# np.savetxt("my_l1_weights.csv", l1_wei)
# l2_wei = sess.run(tf.trainable_variables()[-1])
# np.savetxt("my_l2_weights.csv", l2_wei)

# writer = tf.summary.FileWriter('./my_train.log', sess.graph)

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run(
        [opt_op, obj_func, accuracy, sup_loss, l2_norm, vat_loss],
        # [opt_op, obj_func, accuracy, sup_loss, l2_norm, vat_loss, r_vadv,
        #  r_d, r_dist, logits, r_logit_m, logit_m],
        feed_dict=feed_dict
    )
    # Print training results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "tr_obj=", "{:.5f}".format(outs[1]),
        "tr_acc=", "{:.5f}".format(outs[2]),
        "tr_loss=", "{:.5f}".format(outs[3]),
        "tr_l2=", "{:.5f}".format(outs[4]),
        "tr_vat=", "{:.10f}".format(outs[5])
    )

    # Validation
    feed_dict_val = construct_feed_dict(features, support, y_val, val_mask, placeholders)
    outs_val = sess.run(
        [obj_func, accuracy, sup_loss, l2_norm, vat_loss],
        feed_dict=feed_dict_val
    )
    cost_val.append(outs_val[2])

    # Print validation results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "va_obj=", "{:.5f}".format(outs_val[0]),
        "va_acc=", "{:.5f}".format(outs_val[1]),
        "va_loss=", "{:.5f}".format(outs_val[2]),
        "va_l2=", "{:.5f}".format(outs_val[3]),
        "va_vat=", "{:.10f}".format(outs_val[4])
    )

    # Testing
    feed_dict_tes = construct_feed_dict(features, support, y_test, test_mask,
                                        placeholders)
    outs_tes = sess.run(
        [obj_func, accuracy, sup_loss, l2_norm, vat_loss],
        feed_dict=feed_dict_tes
    )

    # Print testing results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "te_obj=", "{:.5f}".format(outs_tes[0]),
        "te_acc=", "{:.5f}".format(outs_tes[1]),
        "te_loss=", "{:.5f}".format(outs_tes[2]),
        "te_l2=", "{:.5f}".format(outs_tes[3]),
        "te_vat=", "{:.10f}".format(outs_tes[4])
    )

    epoch_duration = time.time() - t
    print('-------', 'time=', "{:.5f}".format(epoch_duration), '------')

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
t_test = time.time()
feed_dict_tes = construct_feed_dict(features, support, y_test, test_mask, placeholders)
outs_tes = sess.run([obj_func, accuracy, logits], feed_dict=feed_dict_tes)
test_cost, test_acc, test_duration = outs_tes[0], outs_tes[1], (time.time() - t_test)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
