__author__ = 'robert'

"""
Adjusted soinn classification
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


def distance(x, y):
    """
    Norm-2 distance between 2 points
    """
    return np.linalg.norm((y - x))


def asoinn(data, age_max, lamb):
    """ A fast soinn function """

    # Initialize 2 nodes
    nodes = data[0:2, :]
    m = np.array([1, 1])
    tem = distance(data[0, :], data[1, :])
    threshold = np.array([tem, tem])
    connection = np.array([[0, 0], [0, 0]])
    age = np.array([[0, 0], [0, 0]])

    value = np.array([0., 0.], dtype=np.float64)
    index = np.array([0, 0], dtype=np.int32)
    sample_size = data.shape[0]
    for i in xrange(2, sample_size):
        # find winner node and runner-up node
        dist = np.sqrt(np.sum((np.matlib.repmat(data[i, :], nodes.shape[0], 1) - nodes) ** 2, axis=1))
        value[0] = np.amin(dist)
        index[0] = np.argmin(dist)
        dist[index[0]] = 1000000
        value[1] = np.amin(dist)
        index[1] = np.argmin(dist)
        # prototype, connection and age update
        if value[0] > threshold[index[0]] or value[1] > threshold[index[1]]:
            # add a new prototype
            nodes = np.concatenate((nodes, np.reshape(data[i, :], (1, -1))), axis=0)
            threshold = np.concatenate((threshold, np.array([1000000])))
            m = np.concatenate((m, np.array([1])))
            connection = np.concatenate((connection, np.zeros((1, connection.shape[1]))), axis=0)
            connection = np.concatenate((connection, np.zeros((connection.shape[0], 1))), axis=1)
            age = np.concatenate((age, np.zeros((1, age.shape[1]))), axis=0)
            age = np.concatenate((age, np.zeros((age.shape[0], 1))), axis=1)
        else:
            # find neighbor nodes of winner nodes
            neighbors = np.nonzero(connection[index[0], :])[0]
            age[index[0], neighbors] += 1
            age[neighbors, index[0]] += 1
            # build connection
            if connection[index[0], index[1]] == 0:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1

            age[index[1], index[0]] = 0
            age[index[0], index[1]] = 0
            neighbors = np.nonzero(connection[index[0], :])[0]
            # adjust the weight of winner node
            m[index[0]] += 1
            nodes[index[0], :] += (1.0 / np.float64(m[index[0]])) * (data[i, :] - nodes[index[0], :])
            if neighbors.shape[0] > 0:
                nodes[neighbors, :] += (1.0 / (100.0 * np.float64(m[index[0]]))) * \
                                        (np.matlib.repmat(data[i, :], neighbors.shape[0], 1) - nodes[neighbors, :])
            # delete the edges whose ages are greater than age_max
            locate = np.where(age[index[0], :] > age_max)[0]
            connection[index[0], locate] = 0
            connection[locate, index[0]] = 0
            age[index[0], locate] = 0
            age[locate, index[0]] = 0

        # update threshold
        if np.count_nonzero(connection[index[0], :]) == 0:
            # no neighbor, the threshold should be the distance between winner node and runner-up node
            threshold[index[0]] = distance(nodes[index[0], :], nodes[index[1], :])
        else:
            # if have neighbors, choose the farthest one
            neighbors = np.nonzero(connection[index[0], :])[0]
            neighbor_distances = np.matlib.repmat(nodes[index[0], :], neighbors.shape[0], 1) - nodes[neighbors, :]
            threshold_winner = np.max(np.sqrt(np.sum(neighbor_distances ** 2, axis=1)))
            threshold[index[0]] = threshold_winner

        if np.count_nonzero(connection[index[1], :]) == 0:
            # no neighbor
            threshold[index[1]] = distance(nodes[index[0], :], nodes[index[1], :])
        else:
            neighbors = np.nonzero(connection[index[1], :])[0]
            neighbor_distances = np.matlib.repmat(nodes[index[1], :], neighbors.shape[0], 1) - nodes[neighbors, :]
            threshold_runner = np.max(np.sqrt(np.sum(neighbor_distances ** 2, axis=1)))
            threshold[index[1]] = threshold_runner

        if (i + 1) % lamb == 0:
            # mean point
            # delete nodes with 0, 1 or 2 neighbors
            # mean_m = np.mean(m)
            neighbors = np.sum(connection, axis=1)
            # neighbor0_set = np.intersect1d(np.where(m < mean_m)[0], np.where(neighbors == 0)[0])
            neighbor0_set = np.where(neighbors == 0)[0]
            # neighbor1_set = np.intersect1d(np.where(m < c1 * mean_m)[0], np.where(neighbors == 1)[0])
            neighbor1_set = np.where(neighbors == 1)[0]
            to_delete = np.union1d(neighbor0_set, neighbor1_set)
            if nodes.shape[0] - to_delete.shape[0] < 2:
                continue
            nodes = np.delete(nodes, to_delete, axis=0)
            threshold = np.delete(threshold, to_delete)
            m = np.delete(m, to_delete)
            connection = np.delete(connection, to_delete, axis=0)
            connection = np.delete(connection, to_delete, axis=1)
            age = np.delete(age, to_delete, axis=0)
            age = np.delete(age, to_delete, axis=1)

    return nodes, connection


def kmeans(data, centroids):
    """ k-means method """
    cluster_changed = True
    samples = data.shape[0]
    k = centroids.shape[0]
    cluster_assignment = np.ones((samples, 1), dtype=np.int32) * -1
    cluster_assignment0 = np.copy(cluster_assignment)
    while cluster_changed:
        cluster_changed = False
        dist_cluster = np.ones((samples, k)) * np.inf
        for i in xrange(k):
            dist_cluster[:, i] = np.sqrt(np.sum((data - np.matlib.repmat(centroids[i, :], samples, 1)) ** 2, axis=1))
        cluster_assignment = np.argmin(dist_cluster, axis=1)
        if not np.array_equal(cluster_assignment, cluster_assignment0):
            cluster_changed = True
        cluster_assignment0 = np.copy(cluster_assignment)
        for i in xrange(k):
            points_in_cluster = data[np.where(cluster_assignment == i)[0], :]
            centroids[i, :] = np.mean(points_in_cluster, axis=0)
    return centroids


def test_kmeans():
    data = np.array([[0.5, 0.5], [0.6, 0.6], [1.0, 1.0], [1.1, 1.1]])
    cent = np.array([[0.45, 0.45], [1.2, 1.2]])
    cent = kmeans(data, cent)


def kNN(inX, data, labels, k):
    """
    k nearby neighbors method
    """
    prototypes_num = data.shape[0]
    k = min(prototypes_num, k)
    dist = np.sqrt(np.sum((np.tile(inX, (prototypes_num, 1)) - data) ** 2, axis=1))
    neighbors = np.argsort(dist)[0:k]
    voted_label = np.argmax(np.bincount(labels[neighbors]))     # labels are expressed by integer 0 ~ ..n
    return voted_label


def test_kNN():
    data = np.array([[0.5, 0.5], [0.6, 0.6], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    inX = np.array([1.4, 1.4])
    k = 2
    input_label = kNN(inX, data, labels, k)
    print "Test kNN: label %d (expected: 0)" % input_label
    pass


def asc_reduce_noise(prototypes, labels, k):
    n = prototypes.shape[0]
    i = 0
    while i < n:
        i += 1
        item = prototypes[0, :]
        item_label = labels[0]
        prototypes = np.delete(prototypes, 0, axis=0)    # delete first, for kNN method
        labels = np.delete(labels, 0)
        knn_label = kNN(item, prototypes, labels, k)
        if knn_label == item_label:
            # this prototype is valid, restored this prototype
            prototypes = np.concatenate((prototypes, item.reshape((1, -1))), axis=0)
            labels = np.concatenate((labels, np.array([item_label])))
    return prototypes, labels


def asc_clean_centers(data, labels, prototypes, prototype_labels):
    prototype_selected_count = np.zeros_like(prototype_labels, dtype=np.int32)
    samples = data.shape[0]
    for i in xrange(samples):
        sample_label = labels[i]
        other_prototypes_index = np.where(prototypes != sample_label)[0]
        other_prototypes = prototypes[other_prototypes_index, :]
        dist = np.sqrt(np.sum((np.tile(data[i, :], (other_prototypes.shape[0], 1)) - other_prototypes) ** 2))
        nearest_prototype_index = np.argmin(dist)
        prototype_selected_count[other_prototypes_index[nearest_prototype_index]] += 1

    to_delete = np.where(prototype_selected_count == 0)[0]
    np.delete(prototypes, to_delete, axis=0)
    np.delete(prototype_labels, to_delete)
    return prototypes, prototype_labels


def asc(data, labels, age_max, lamb, k):
    clusters = np.amax(labels) + 1
    for i in xrange(clusters):
        item_index_in_cluster = np.where(labels == i)[0]
        data_in_cluster = data[item_index_in_cluster, :]
        prototypes_cluster, _ = asoinn(data_in_cluster, age_max, lamb)
        prototypes_cluster = kmeans(data_in_cluster, prototypes_cluster)
        prototype_cluster_labels = np.ones(prototypes_cluster.shape[0], dtype=np.int32) * i
        if i == 0:
            prototypes = prototypes_cluster
            prototypes_labels = prototype_cluster_labels
        else:
            prototypes = np.concatenate((prototypes, prototypes_cluster), axis=0)
            prototypes_labels = np.concatenate((prototypes_labels, prototype_cluster_labels))

    prototypes, prototypes_labels = asc_reduce_noise(prototypes, prototypes_labels, k)
    prototypes, prototypes_labels = asc_clean_centers(data, labels, prototypes, prototypes_labels)
    return prototypes, prototypes_labels


def test_asc():
    mean1 = [0.5, 0.5]
    mean2 = [3.0, 0.5]
    conv1 = [[1, 0], [0, 1]]
    conv2 = [[1.5, 0], [0, 1.2]]
    samples = 2000
    class1_data = np.random.multivariate_normal(mean1, conv1, samples)
    class1_label = np.ones(samples, dtype=np.int32) * 0
    class2_data = np.random.multivariate_normal(mean2, conv2, samples)
    class2_label = np.ones(samples, dtype=np.int32) * 1
    train_data = np.concatenate((class1_data, class2_data))
    train_label = np.concatenate((class1_label, class2_label))

    prototypes, labels = asc(train_data, train_label, 20, 20, 3)
    plt.switch_backend('Qt4Agg')
    plt.figure(1)
    plt.hold(True)
    plt.plot(class1_data[:, 0], class1_data[:, 1], 'r*')
    plt.plot(class2_data[:, 0], class2_data[:, 1], 'b.')
    plt.figure(2)
    plt.hold(True)
    for i in xrange(labels.shape[0]):
        if labels[i] == 0:
            plt.plot(prototypes[i, 0], prototypes[i, 1], 'r*')
        else:
            plt.plot(prototypes[i, 0], prototypes[i, 1], 'b.')
    plt.hold(False)
    plt.show()


if __name__ == '__main__':
    test_asc()
