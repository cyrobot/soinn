__author__ = 'robert'

import time
# import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import fast_soinn


plt.switch_backend('Qt4Agg')

start_time = time.time()
train_data_file = 'soinn_demo_train.mat'
train_data = sio.loadmat(train_data_file)
train = train_data['train']
end_time = time.time()
print 'loading train data executes %s seconds' % (end_time - start_time)

plt.figure(1)
plt.plot(train[:, 0], train[:, 1], 'bo')

start_time = time.time()
nodes, connection, classes = fast_soinn.fast_soinn(train, 50, 100, 1.5, 0.001)
end_time = time.time()
print 'classes is %s' % classes
print 'soinn execute %s seconds' % (end_time - start_time)

plt.figure(2)
plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

plt.hold(True)
for i in xrange(0, nodes.shape[0]):
    for j in xrange(0, nodes.shape[0]):
        if connection[i, j] != 0:
            plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 'b-')
            pass
plt.hold(False)

plt.show()
