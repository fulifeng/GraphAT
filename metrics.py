import copy
import numpy as np
import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def np_masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = np.equal(np.argmax(preds, 1), np.argmax(labels, 1))
    accuracy_all = correct_prediction.astype(np.float32)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)


def np_masked_accuracy_by_degree(preds, labels, mask, degrees, groups):
    accuracies = []
    n = len(mask)
    for group in groups:
        cur_mask = copy.copy(mask)
        evaluated_nodes = 0
        for i in range(n):
            if group[0] <= degrees[i] <= group[1] and mask[i]:
                evaluated_nodes += 1
                cur_mask[i] = True
            else:
                cur_mask[i] = False
        cur_acc = np_masked_accuracy(preds, labels, cur_mask)
        print(evaluated_nodes, cur_acc)
        accuracies.append(cur_acc)
    return accuracies


def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def my_kld_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=q)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=q_logit, labels=q)
    return tf.reduce_mean(cross_entropy - entropy)


def my_kld_with_logit_with_mask(q_logit, p_logit, mask):
    q = tf.nn.softmax(q_logit)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=q)
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=q_logit, labels=q)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    return tf.reduce_mean((cross_entropy - entropy) * mask)


def my_softmax_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=q))


def my_neighbor_kld_with_logit(neighbor_logits, p_logit):
    dist = 0
    for i in range(len(neighbor_logits)):
        dist += my_kld_with_logit(neighbor_logits[i], p_logit)
    return dist


def my_neighbor_softmax_with_logit(neighbor_logits, p_logit):
    dist = 0
    for i in range(len(neighbor_logits)):
        dist += my_softmax_with_logit(neighbor_logits[i], p_logit)
    return dist