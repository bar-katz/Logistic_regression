import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.legend_handler import HandlerLine2D


def main():
    num_of_samples = 100
    num_of_groups = 3
    num_of_epochs = 30
    eta = 0.1

    train_data_y = np.asarray([0] * num_of_samples + [1] * num_of_samples + [2] * num_of_samples)
    train_data_x = np.concatenate([np.random.normal(2, 1, num_of_samples),
                                   np.random.normal(4, 1, num_of_samples),
                                   np.random.normal(6, 1, num_of_samples)], axis=0)

    w = np.zeros(num_of_groups)
    b = np.zeros(num_of_groups)

    for e in range(num_of_epochs):
        # shuffle samples
        s = np.arange(train_data_x.shape[0])
        np.random.shuffle(s)
        train_data_x = train_data_x[s]
        train_data_y = train_data_y[s]

        for x, y in zip(train_data_x, train_data_y):
            y_hat = np.argmax(softmax(np.dot(w, x) + b))
            if y_hat != y:
                # loss = loss_neg_log_likelihood(w, b, train_data_x, train_data_y)
                for i in range(3):
                    z = softmax(np.dot(w, x) + b)
                    if i == y:
                        w[i] += (-eta * (x * z[i] - x))
                        b[i] += (-eta * (-1 + z[i]))
                    else:
                        w[i] += (-eta * (x * z[i]))
                        b[i] += (-eta * (z[i]))

    all_x = list()
    normal_y = list()
    modal_y = list()

    for val in range(0, 101):
        x = val / 10.0
        all_x.append(x)
        normal_y.append(normal(2, x) / (normal(2, x) + normal(4, x) + normal(6, x)))
        modal_y.append(softmax(np.dot(w, x) + b)[0])

    plt.axis([0, 10, 0, 2])
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    normal_graph, = plt.plot(all_x, normal_y, 'r--', label="normal distribution")

    plt.plot(all_x, modal_y, label="logistic regression")

    plt.legend(handler_map={normal_graph: HandlerLine2D(numpoints=4)})
    plt.show()


def normal(m, x):
    return (1.0 / math.sqrt(2 * math.pi)) * np.exp((-(x - m) ** 2) / 2)


def softmax(w):
    e = np.exp(np.array(w) - np.max(w))
    return e / np.sum(e)


def loss_neg_log_likelihood(w, b, train_x, train_y):
    loss = 0
    for i, x, y in zip(range(len(train_x)), train_x, train_y):
        loss += np.log(softmax(np.dot(w[i, :], x) + b[i])[y])

    return -loss


if __name__ == "__main__":
    main()
