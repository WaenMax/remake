import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
import pickle
import os


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_mp_matrices(path, mp_names):
    mp_dict = {}
    for mp_name in mp_names:
        mp_dict[mp_name] = sp.load_npz(os.path.join(path, "mp", mp_name + ".npz")).tocoo().astype(np.float32)
    return mp_dict


def build_mp_statistics(mp_dict, target_nodes_num):
    mp_stats = {}
    for mp_name, mp_matrix in mp_dict.items():
        edge_num = int(mp_matrix.nnz)
        if edge_num == 0:
            semantic_density = 0.0
            mean_weight = 0.0
        else:
            mean_weight = float(np.asarray(mp_matrix.data, dtype=np.float32).mean())
            semantic_density = mean_weight

        edge_node_ratio = edge_num / float(target_nodes_num)
        structural_sparsity = 1.0 / (edge_node_ratio + 1e-12)
        mp_stats[mp_name] = {
            "edge_num": edge_num,
            "node_num": int(target_nodes_num),
            "mean_weight": mean_weight,
            "semantic_density": semantic_density,
            "edge_node_ratio": edge_node_ratio,
            "structural_sparsity": structural_sparsity,
        }
    return mp_stats


def build_mp_metadata(path, core_mp_names, aux_mp_names, target_nodes_num):
    all_mp_names = core_mp_names + aux_mp_names
    mp_matrices = load_mp_matrices(path, all_mp_names)
    mp_stats = build_mp_statistics(mp_matrices, target_nodes_num)

    core_mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_matrices[mp]) for mp in core_mp_names}
    aux_mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_matrices[mp]) for mp in aux_mp_names}
    all_mp_dict = {mp: sparse_mx_to_torch_sparse_tensor(mp_matrices[mp]) for mp in all_mp_names}

    return {
        "core_mp_dict": core_mp_dict,
        "aux_mp_dict": aux_mp_dict,
        "all_mp_dict": all_mp_dict,
        "core_mp_names": core_mp_names,
        "aux_mp_names": aux_mp_names,
        "all_mp_names": all_mp_names,
        "mp_stats": mp_stats,
    }


def load_acm(ratio, type_num, core_mp_names=None, aux_mp_names=None):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_s = sp.load_npz(path + "s_feat.npz")
   
    core_mp_names = core_mp_names or ["pap", "psp"]
    aux_mp_names = aux_mp_names or ["apa", "pspap", "sps"]
    mp_metadata = build_mp_metadata(path, core_mp_names, aux_mp_names, target_nodes_num=3025)
    mp_dict = mp_metadata["core_mp_dict"]

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    feat_dic = {'p':feat_p, 'a':feat_a, 's':feat_s}
    type_range = {'p': range(0, 3025), 
                  'a': range(3025, 3025+5912), 
                  's': range(3025+5912, 3025+5912+57)}

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return feat_dic, type_range, mp_dict, label, train, val, test, mp_metadata


def load_dblp(ratio, type_num, core_mp_names=None, aux_mp_names=None):
    # The order of node types: 0 a 1 p 2 c
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_c = sp.load_npz(path + "c_feat.npz")

    core_mp_names = core_mp_names or ["apa", "apcpa"]
    aux_mp_names = aux_mp_names or ["pap", "pcp", "cpc"]
    mp_metadata = build_mp_metadata(path, core_mp_names, aux_mp_names, target_nodes_num=4057)
    mp_dict = mp_metadata["core_mp_dict"]

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_c = th.FloatTensor(preprocess_features(feat_c))

    feat_dic = {'a':feat_a, 'p':feat_p, 'c':feat_c}
    type_range = {'a': range(0, 4057), 
                  'p': range(4057, 4057+14328), 
                  'c': range(4057+14328, 4057+14328+20),}

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return feat_dic, type_range, mp_dict, label, train, val, test, mp_metadata


def load_imdb(ratio, type_num, core_mp_names=None, aux_mp_names=None):
    # The order of node types: 0 m 1 d 2 a
    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_d = sp.load_npz(path + "d_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")

    core_mp_names = core_mp_names or ["mam", "mdm"]
    aux_mp_names = aux_mp_names or ["ama", "dmd", "mamdm"]
    mp_metadata = build_mp_metadata(path, core_mp_names, aux_mp_names, target_nodes_num=4661)
    mp_dict = mp_metadata["core_mp_dict"]
    
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_dic = {'m':feat_m, 'd':feat_d, 'a':feat_a}
    type_range = {'m': range(0, 4661), 
                  'd': range(4661, 4661+2270), 
                  'a': range(4661+2270, 4661+2270+5841)}

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return feat_dic, type_range, mp_dict, label, train, val, test, mp_metadata


def load_data(dataset, ratio, type_num, core_mp_names=None, aux_mp_names=None):
    if dataset == "acm":
        data = load_acm(ratio, type_num, core_mp_names, aux_mp_names)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num, core_mp_names, aux_mp_names)
    elif dataset == "imdb":
        data = load_imdb(ratio, type_num, core_mp_names, aux_mp_names)
    return data
