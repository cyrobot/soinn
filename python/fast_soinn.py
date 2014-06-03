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
    nodes_point = np.array([0.0, 0.0])
    nodes_point0 = np.array([0.0, 0.0])
    nodes_point_valid_count = np.array([0, 0])
    nodes_density = np.array([0.0, 0.0])
    nodes_class_id = np.array([1, 0])

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
            nodes_point = np.concatenate((nodes_point, np.array([0.0])))
            nodes_point0 = np.concatenate((nodes_point0, np.array([0.0])))
            nodes_point_valid_count = np.concatenate((nodes_point_valid_count, np.array([0])))
            nodes_density = np.concatenate((nodes_density, np.array([0.0])))
            nodes_class_id = np.concatenate((nodes_class_id, np.array([np.max(nodes_class_id) + 1])))
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
            if nodes_class_id[index[0]] == 0 or nodes_class_id[index[1]] == 0:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1
            elif nodes_class_id[index[0]] == nodes_class_id[index[1]]:
                connection[index[0], index[1]] = 1
                connection[index[1], index[0]] = 1
                nodes_class_id[index[1]] = nodes_class_id[index[0]]
            elif nodes_class_id[index[0]] != nodes_class_id[index[1]]:
                nodes_class_as_winner = (np.where(nodes_class_id == nodes_class_id[index[0]])[0])
                mean_density_class_winner = np.sum(nodes_density[nodes_class_as_winner]) / \
                                            np.float64(nodes_class_as_winner.shape[0])
                max_density_class_winner = np.max(nodes_density[nodes_class_as_winner])
                if 2.0 * mean_density_class_winner >= max_density_class_winner:
                    alpha_winner = 0.0
                elif 3.0 * mean_density_class_winner >= max_density_class_winner:
                    alpha_winner = 0.5
                else:
                    alpha_winner = 1.0
                winner_condition = min(nodes_density[index[0]], nodes_density[index[1]]) >= \
                                   (alpha_winner * max_density_class_winner)

                nodes_class_as_runner = (np.where(nodes_class_id == nodes_class_id[index[1]])[0])
                mean_density_class_runner = np.sum(nodes_density[nodes_class_as_runner]) / \
                                            np.float64(nodes_class_as_runner.shape[0])
                max_density_class_runner = np.max(nodes_density[nodes_class_as_runner])
                if 2.0 * mean_density_class_runner >= max_density_class_runner:
                    alpha_runner = 0.0
                elif 3.0 * mean_density_class_runner >= max_density_class_runner:
                    alpha_runner = 0.5
                else:
                    alpha_runner = 1.0
                runner_condition = min(nodes_density[index[0]], nodes_density[index[1]]) > \
                                   (alpha_runner * max_density_class_runner)

                if winner_condition or runner_condition:
                    connection[index[0], index[1]] = 1
                    connection[index[1], index[0]] = 1
                    winner_runner_combine_class = min(nodes_class_id[index[0]], nodes_class_id[index[1]])
                    nodes_class_id[nodes_class_as_winner] = winner_runner_combine_class
                else:
                    connection[index[0], index[1]] = 0
                    connection[index[1], index[0]] = 0
                pass

            if connection[index[0], index[1]] == 1:
                nodes_class_id[[index[0], index[1]]] = min(nodes_class_id[index[0]], nodes_class_id[index[1]])

            age[index[1], index[0]] = 0
            age[index[0], index[1]] = 0
            neighbors = np.nonzero(connection[index[0], :])[0]
            # calculate the 'point'
            if neighbors.shape[0] > 0:
                winner_neighbor_diff = np.matlib.repmat(nodes[index[0], :], neighbors.shape[0], 1) - nodes[neighbors, :]
                winner_mean_distance = (1.0 / np.float64(neighbors.shape[0])) * \
                                        np.sum(np.sqrt(np.sum(winner_neighbor_diff ** 2, axis=1)))
                nodes_point[index[0]] += 1.0 / ((1.0 + winner_mean_distance) ** 2)
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
            nodes_point_sum_period = nodes_point - nodes_point0
            nodes_point_valid_node = np.nonzero(nodes_point_sum_period)[0]
            nodes_point_valid_count[nodes_point_valid_node] += 1
            for j in xrange(0, nodes_density.shape[0]):
                if nodes_point_valid_count[j] > 0:
                    nodes_density[j] = nodes_point[j] / nodes_point_valid_count[j]

            nodes_point0 = np.copy(nodes_point)

            # delete nodes with 0, 1 or 2 neighbors
            # mean_m = np.mean(m)
            mean_density = np.mean(nodes_density)
            neighbors = np.sum(connection, axis=1)
            # neighbor0_set = np.intersect1d(np.where(m < mean_m)[0], np.where(neighbors == 0)[0])
            neighbor0_set = np.where(neighbors == 0)[0]
            neighbor1_set = np.intersect1d(np.where(nodes_density < c * mean_density)[0], np.where(neighbors == 1)[0])
            neighbor2_set = np.intersect1d(np.where(nodes_density < 0.001 * mean_density)[0],
                                           np.where(neighbors == 2)[0])
            # neighbor1_set = np.where(neighbors == 1)[0]
            to_delete = np.union1d(neighbor0_set, neighbor1_set)
            to_delete = np.union1d(to_delete, neighbor2_set)
            nodes = np.delete(nodes, to_delete, axis=0)
            threshold = np.delete(threshold, to_delete)
            m = np.delete(m, to_delete)
            nodes_point = np.delete(nodes_point, to_delete)
            nodes_point0 = np.delete(nodes_point0, to_delete)
            nodes_point_valid_count = np.delete(nodes_point_valid_count, to_delete)
            nodes_density = np.delete(nodes_density, to_delete)
            nodes_class_id = np.delete(nodes_class_id, to_delete)
            connection = np.delete(connection, to_delete, axis=0)
            connection = np.delete(connection, to_delete, axis=1)
            age = np.delete(age, to_delete, axis=0)
            age = np.delete(age, to_delete, axis=1)

    return nodes, connection, nodes_class_id
