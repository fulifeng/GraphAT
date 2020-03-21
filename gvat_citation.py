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
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_float('epsilon', 1.0, "norm length for (virtual) adversarial training ")
flags.DEFINE_float('epsilon_graph', 1.0, "norm length for graph adversarial training ")
flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
flags.DEFINE_float('xi', 1e-4, "small constant for finite difference")
flags.DEFINE_float('alpha', 1.0, "Weight for VAT loss")
flags.DEFINE_float('beta', 1.0, "Weight for GAT loss")
flags.DEFINE_bool('mask_vat', False, 'calculate vat loss only on unlabeled data.')
flags.DEFINE_bool('reload', False, 'reload parameter.')
flags.DEFINE_string('model_path', './model/vat/cora/model', 'path to reload model.')
flags.DEFINE_string('model_save_path', './model/vat/cora/model', 'path to save model.')
flags.DEFINE_integer('num_neighbors', 1, "the number of sampled neighbors")
flags.DEFINE_bool('vat_loss', False, 'Include VAT loss.')
flags.DEFINE_bool('gat_loss', False, 'Include GAT loss.')
flags.DEFINE_bool('emb_all', False, 'Include GAT loss.')
flags.DEFINE_string('sampling', 'uniform', 'Strategy to sample neighbors.')


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# there is a confusing point, the following line will print the default value
# of parameters, but will print the input value after the first call of FLAG
# values

# generate_random_splits(y_train, y_val, y_test, train_mask, val_mask,
#                        test_mask, FLAGS.dataset, repeats=5)
print(FLAGS.flag_values_dict())

col_indices = get_col_indices(adj)
col_distributions = get_sampling_probability(col_indices, FLAGS.sampling,
                                             FLAGS.dataset)

# Some preprocessing
if 'nell.0' in FLAGS.dataset:
    features = load_embedding('trans', FLAGS.dataset, FLAGS.emb_all)
else:
    features = preprocess_features(features)

if FLAGS.model == 'gcn':
    # support = [preprocess_adj(adj)]
    support = preprocess_adj(adj)
    num_supports = 1
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    # 'neighbor_features': [tf.placeholder(tf.float32) for _ in range(FLAGS.num_neighbors)],
    'neighbor_ids': [tf.placeholder(tf.int32) for _ in range(FLAGS.num_neighbors)],
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
    # vat_loss = kl_divergence_with_logit(logit_p, logit_m)
    if FLAGS.mask_vat:
        vat_loss = my_kld_with_logit_with_mask(logit_p, logit_m,
                                               placeholders['labels_mask'])
    else:
        vat_loss = my_kld_with_logit(logit_p, logit_m)
    return tf.identity(vat_loss, name=name)
    # return tf.identity(vat_loss, name=name), r_vadv, r_d, r_dist, r_logit_m, logit_m


def generate_graph_adversarial_perturbation(x, logits, neighbor_logits,
                                            is_training=True):
    dist = my_neighbor_kld_with_logit(neighbor_logits, logits)
    # dist = my_neighbor_softmax_with_logit(neighbor_logits, logits)
    grad = tf.gradients(dist, [x], aggregation_method=2)[0]
    d = tf.stop_gradient(grad)
    return FLAGS.epsilon_graph * tf.nn.l2_normalize(d, axis=1)
    # return FLAGS.epsilon * tf.nn.l2_normalize(d, axis=1), d, dist, logit_m


def graph_adversarial_loss(x, logits, is_training=True, name="gat_loss"):
    neighbor_logits = list()
    for i in range(FLAGS.num_neighbors):
        neighbor_logit = tf.gather(logits, placeholders['neighbor_ids'][i])
        neighbor_logit = tf.stop_gradient(neighbor_logit)
        neighbor_logits.append(neighbor_logit)
    #
    r_gadv = generate_graph_adversarial_perturbation(
        x, logits, neighbor_logits, is_training=is_training
    )
    # r_vadv, r_d, r_dist, r_logit_m = generate_virtual_adversarial_perturbation(
    #     x, logits, is_training=is_training
    # )
    logit_m = logit(x + r_gadv, is_training=is_training)
    gat_loss = my_neighbor_kld_with_logit(neighbor_logits, logit_m)
    # gat_loss = my_neighbor_softmax_with_logit(neighbor_logits, logit_m)

    # return tf.identity(gat_loss, name=name)
    return tf.identity(gat_loss, name=name), logit_m
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
        # vat_loss = virtual_adversarial_loss(placeholders['features'], logits)
        # # vat_loss = tf.identity(0.0, name='vat_loss')
        # gat_loss = graph_adversarial_loss(placeholders['features'], logits)

        # vat loss
        if FLAGS.vat_loss:
            vat_loss = virtual_adversarial_loss(placeholders['features'], logits)
        else:
            vat_loss = tf.identity(0.0, name='vat_loss')

        # gat loss
        if FLAGS.gat_loss:
            # gat_loss = graph_adversarial_loss(placeholders['features'], logits)
            gat_loss, logits_m = graph_adversarial_loss(placeholders['features'], logits)
        else:
            gat_loss = tf.identity(0.0, name='gat_loss')

        obj_func = sup_loss + FLAGS.weight_decay * l2_norm + \
                   FLAGS.alpha * vat_loss + FLAGS.beta * gat_loss
    # obj_func = sup_loss + FLAGS.weight_decay * l2_norm

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    opt_op = optimizer.minimize(obj_func)

    accuracy = masked_accuracy(logits, placeholders['labels'],
                               placeholders['labels_mask'])


# Initialize session
sess = tf.Session()

# writer = tf.summary.FileWriter('./gvat' + str(FLAGS.num_neighbors) + '.log',
#                                sess.graph)

# # Define model evaluation function
# def evaluate(features, support, labels, mask, placeholders):
#     t_test = time.time()
#     feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
#     outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
#     return outs_val[0], outs_val[1], (time.time() - t_test)


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

cost_val = []


# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    # neighbors_features = sample_neighbors(col_indices, features, 3)
    neighbors_ids = sample_neighbors_id(col_indices, FLAGS.num_neighbors, col_distributions)
    feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                    placeholders, neighbor_ids=neighbors_ids)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # init_vat_loss = sess.run([vat_loss], feed_dict=feed_dict)
    # print('tr_vat=', "{:.15f}".format(init_vat_loss[0]))

    # Training step
    outs = sess.run(
        [opt_op, obj_func, accuracy, sup_loss, l2_norm, vat_loss, gat_loss],
        # [opt_op, obj_func, accuracy, sup_loss, l2_norm, vat_loss, r_vadv,
        #  r_d, r_dist, logits, r_logit_m, logit_m],
        feed_dict=feed_dict
    )
    # np.savetxt('logits.csv', my_softmax(outs[-3]), fmt='%.5f')
    # np.savetxt('vat_logits.csv', my_softmax(outs[-1]), fmt='%.5f')
    # Print training results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "tr_obj=", "{:.4f}".format(outs[1]),
        "tr_acc=", "{:.4f}".format(outs[2]),
        "tr_loss=", "{:.4f}".format(outs[3]),
        "tr_l2=", "{:.4f}".format(outs[4]),
        "tr_vat=", "{:.6f}".format(outs[5]),
        "tr_gat=", "{:.6f}".format(outs[6])
    )

    # Validation
    feed_dict_val = construct_feed_dict(
        features, support, y_val, val_mask, placeholders,
        neighbor_ids=neighbors_ids
    )
    outs_val = sess.run(
        [obj_func, accuracy, sup_loss, l2_norm, vat_loss, gat_loss, logits],
        feed_dict=feed_dict_val
    )
    # cost, acc, duration = outs_val[0], outs_val[1], (time.time() - t_test)
    cost_val.append(outs_val[2])

    # Print validation results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "va_obj=", "{:.4f}".format(outs_val[0]),
        "va_acc=", "{:.4f}".format(outs_val[1]),
        "va_loss=", "{:.4f}".format(outs_val[2]),
        "va_l2=", "{:.4f}".format(outs_val[3]),
        "va_vat=", "{:.6f}".format(outs_val[4]),
        "va_gat=", "{:.6f}".format(outs_val[5])
    )
    # print("TTTT:", '%04d' % (epoch + 1), "te_acc=", "{:.4f}".format(
    #     np_masked_accuracy(outs_val[6], y_test, test_mask)
    # ))

    # Testing
    # t_test = time.time()
    feed_dict_tes = construct_feed_dict(
        features, support, y_test, test_mask, placeholders,
        neighbor_ids=neighbors_ids
    )
    outs_tes = sess.run(
        [obj_func, accuracy, sup_loss, l2_norm, vat_loss, gat_loss],
        feed_dict=feed_dict_tes
    )
    
    # Print testing results
    print(
        "Epoch:", '%04d' % (epoch + 1),
        "te_obj=", "{:.4f}".format(outs_tes[0]),
        "te_acc=", "{:.4f}".format(outs_tes[1]),
        "te_loss=", "{:.4f}".format(outs_tes[2]),
        "te_l2=", "{:.4f}".format(outs_tes[3]),
        "te_vat=", "{:.6f}".format(outs_tes[4]),
        "te_vat=", "{:.6f}".format(outs_tes[5])
    )
    epoch_duration = time.time() - t
    print('-------', 'time=', "{:.5f}".format(epoch_duration), '------')

    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
    #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
    #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(duration))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# # Testing
t_test = time.time()
neighbors_ids = sample_neighbors_id(col_indices, FLAGS.num_neighbors)
feed_dict_tes = construct_feed_dict(features, support, y_test, test_mask,
                                    placeholders, neighbor_ids=neighbors_ids)
outs_tes = sess.run([obj_func, accuracy, logits, logits_m], feed_dict=feed_dict_tes)
test_cost, test_acc, test_duration = outs_tes[0], outs_tes[1], (time.time() - t_test)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
