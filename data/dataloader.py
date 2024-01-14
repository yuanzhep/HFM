"""
@author: yz
01/09/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar_10_data(data_dir, negatives=False):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names

if __name__ == "__main__":
    cifar_10_dir = "./data/cifar10"
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        load_cifar_10_data(cifar_10_dir)

    # print("Train data: ", train_data.shape)
    # print("Train filenames: ", train_filenames.shape)
    # print("Train labels: ", train_labels.shape)
    # print("Test data: ", test_data.shape)
    # print("Test filenames: ", test_filenames.shape)
    # print("Test labels: ", test_labels.shape)
    # print("Label names: ", label_names.shape)

    # num_plot = 5
    # f, ax = plt.subplots(num_plot, num_plot)
    # for m in range(num_plot):
    #     for n in range(num_plot):
    #         idx = np.random.randint(0, train_data.shape[0])
    #         ax[m, n].imshow(train_data[idx])
    #         ax[m, n].get_xaxis().set_visible(False)
    #         ax[m, n].get_yaxis().set_visible(False)
    # f.subplots_adjust(hspace=0.1)
    # f.subplots_adjust(wspace=0)
    # plt.show()
