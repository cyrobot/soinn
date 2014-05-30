__author__ = 'robert'

import numpy as np
import numpy.matlib


def distance(x, y):
    return np.linalg.norm((y - x))


def fast_soinn(data, age_max, lamb, c):
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
            # if no connection between winner node and runner-up node, connect with each other
            if connection[index[0], index[1]] == 0:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1
                age[index[0], index[1]] = 1
                age[index[1], index[0]] = 1
            else:
                age[index[0], index[1]] = 1
                age[index[1], index[0]] = 1
            # find neighbor nodes of winner nodes
            neighbors = np.nonzero(connection[index[0], :])[0]
            age[index[0], neighbors] += 1
            age[neighbors, index[0]] += 1
            # delete noise nodes with too big ages
            locate = np.where(age[index[0], :] > age_max)[0]
            connection[index[0], locate] = 0
            connection[locate, index[0]] = 0
            age[index[0], locate] = 0
            age[locate, index[0]] = 0
            m[index[0]] += 1
            # adjust the weight of winner node
            nodes[index[0], :] += (1.0 / np.float64(m[index[0]])) * (data[i, :] - nodes[index[0], :])

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
            mean_m = np.mean(m)
            neighbors = np.sum(connection, axis=1)
            # neighbor0_set = np.intersect1d(np.where(m < mean_m)[0], np.where(neighbors == 0)[0])
            neighbor0_set = np.where(neighbors == 0)[0]
            # neighbor1_set = np.intersect1d(np.where(m < c * mean_m)[0], np.where(neighbors == 1)[0])
            neighbor1_set = np.where(neighbors == 1)[0]
            to_delete = np.union1d(neighbor0_set, neighbor1_set)
            nodes = np.delete(nodes, to_delete, axis=0)
            threshold = np.delete(threshold, to_delete)
            m = np.delete(m, to_delete)
            connection = np.delete(connection, to_delete, axis=0)
            connection = np.delete(connection, to_delete, axis=1)
            age = np.delete(age, to_delete, axis=0)
            age = np.delete(age, to_delete, axis=1)

    return nodes, connection
