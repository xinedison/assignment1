import numpy as np

from data import gen_datas, data_iter
import autodiff as ad

## get dataset
feature_dim = 1
num_example = 10000
batch_size = 10
epoches = 4
lr = 0.1

features, labels, true_w, true_b = gen_datas(feature_dim, num_example)


## define model
x = ad.placeholder(name = "x")
label = ad.placeholder(name = "label")
weight = ad.Variable(name = "weight")
bias = ad.Variable(name = "bias")

def get_logistic_model(x, weight, bias):
    y = 1 / (1+ad.exp_op(-1 * (ad.matmul_op(x, weight, trans_B=True)+bias)))
    #y = 1 / (1+ad.exp_op(-1 * (ad.mul_op(x, weight)+bias)))
    return y

def loss_function(y, label):
    return y - label

model = get_logistic_model(x, weight, bias)
loss = loss_function(model, label)

## train loop
def sgd(params, params_grads, lr, batch_size):
    '''
    learning algorithm
    '''
    out_params = []
    for param, grad in zip(params, params_grads):
        param = param - lr * grad / batch_size
        out_params.append(param)
    return out_params
        

grad_w, grad_b = ad.gradients(loss, [weight, bias])
loss_total = ad.reduce_sum_op(loss)
executor = ad.Executor([loss_total, grad_w, grad_b])

weight_val = np.zeros((1,feature_dim))
bias_val = np.zeros((1))

#weight_val = true_w.reshape((1,feature_dim))
#bias_val = true_b
for epoch in range(epoches):
    for batch_idx, feat_val, label_val in data_iter(batch_size, features, labels):
        label_val = label_val.reshape((batch_size, 1))
        loss_val, grad_w_val, grad_b_val = executor.run(feed_dict = {x : feat_val, label : label_val, weight : weight_val, bias : bias_val})

        if batch_idx % 100 == 0:
            print("[Epoch {}, Batch {}] loss : {}".format(epoch, batch_idx, loss_val))

        params = [weight_val, bias_val]
        params_grads = [grad_w_val.T, np.sum(grad_b_val)]
        weight_val, bias_val = sgd(params, params_grads, lr, batch_size)

print('learned weight : ', weight_val, ' bias : ', bias_val)
print('real weight : ', true_w, ' bias : ', true_b)




        




