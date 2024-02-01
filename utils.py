"""
    Some handy functions for pytroch model training ...
"""
import torch
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import logging
import math


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def construct_user_relation_graph_via_item(round_user_params, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def construct_user_relation_graph_via_user(round_user_params, latent_dim, similarity_metric):
    # prepare the user embedding array.
    user_embedding = np.zeros((len(round_user_params), latent_dim), dtype='float32')
    for user in round_user_params.keys():
        user_embedding[user] = copy.deepcopy(round_user_params[user]['embedding_user.weight'].numpy())
    # construct the user relation graph.
    adj = pairwise_distances(user_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1/high_num
            else:
                topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph


def MP_on_graph(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num*latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item.weight'].numpy().flatten()

    # aggregate item embedding via message passing.
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding)
    for layer in range(layers-1):
        aggregated_item_embedding = np.matmul(topk_user_relation_graph, aggregated_item_embedding)

    # reconstruct item embedding.
    item_embedding_dict = {}
    for user in round_user_params.keys():
        item_embedding_dict[user] = torch.from_numpy(aggregated_item_embedding[user].reshape(item_num, latent_dim))
    item_embedding_dict['global'] = sum(item_embedding_dict.values())/len(round_user_params)
    return item_embedding_dict


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == 'embedding_item.weight':
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss