import numpy as np
import random

def gen_datas(feature_dim, num_examples):
    true_w = np.random.random(feature_dim)
    #true_b = np.random.random(1)[0]
    true_b = 0
    features = np.random.normal(scale=1, size=(num_examples, feature_dim))
    w_x = 0
    for i in range(feature_dim):
        w_x += true_w[i] * features[:, i] 
    labels = w_x + true_b
    labels = 1 / (1+np.exp(-1*labels))
    #labels += np.random.normal(scale=0.01, size=labels.shape)

    return features, labels, true_w, true_b


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield i, features[j], labels[j]  

if __name__ == '__main__':
    f, l, true_w, true_b = gen_datas(3, 10)
    for x,y in data_iter(2, f, l):
        print(x, y)
