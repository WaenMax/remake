import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop > 0:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp, list(beta.data.cpu().numpy())


class Gat_layer(nn.Module):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, hidden_dim, attn_drop, head_num = 4, bias=True):
        super(Gat_layer, self).__init__()
        
        self.head_num = head_num
        self.num_out_features = hidden_dim // head_num

        self.linear_proj = nn.Linear(hidden_dim, head_num * hidden_dim // head_num, bias=True)
        nn.init.xavier_normal_(self.linear_proj.weight, gain=1.414)

        # (1, 1, h_dim)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, head_num, self.num_out_features), requires_grad=True)
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, head_num, self.num_out_features), requires_grad=True)
        nn.init.xavier_normal_(self.scoring_fn_target.data, gain=1.414)
        nn.init.xavier_normal_(self.scoring_fn_source.data, gain=1.414)

        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.leakyReLU = nn.LeakyReLU()
        self.activation = nn.PReLU()

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(0, src_nodes_index)
        scores_target = scores_target.index_select(0, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(0, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) 
        # i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[0] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(0, trg_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[0] = num_of_nodes  # shape = (N, NH, out_dim)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, out_dim)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, out_dim) -> (N, NH, out_dim)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def forward(self, in_nodes_features, edge_index):
        # (N, h_dim) -> (N, NH, out_dim)
        # nodes_features_proj = torch.unsqueeze(in_nodes_features, 1)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.head_num, self.num_out_features)
        # nodes_features_proj = self.attn_drop(nodes_features_proj)

        # (N, NH)
        scoring_fn_source_curr = self.attn_drop(self.scoring_fn_source)
        scoring_fn_target_curr = self.attn_drop(self.scoring_fn_target)
        scores_source = (nodes_features_proj * scoring_fn_source_curr).sum(dim=-1)
        scores_target = (nodes_features_proj * scoring_fn_target_curr).sum(dim=-1)

        # (E, NH), (E, NH), (E, NH, out_dim)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = \
                self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        # (E, NH)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        
        num_of_nodes = in_nodes_features.shape[0]
        # (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_of_nodes)
        # attentions_per_edge = self.attn_drop(attentions_per_edge)

        # (E, NH, out_dim)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # shape = (N, NH, out_dim)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        # shape = (N, NH, out_dim) -> (N, NH * out_dim)
        out_nodes_features = out_nodes_features.view(-1, self.head_num * self.num_out_features)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class Mp_attn_encoder(nn.Module):
    def __init__(self, h):
        super(Mp_attn_encoder, self).__init__()

        self.node_att = nn.ModuleDict({mp: Gat_layer(h.hidden_dim, h.attn_drop) for mp in h.mp_name})
        self.mp_att = Attention(h.hidden_dim, h.attn_drop)

        self.mp_name = h.mp_name
        
        self.nei_mask = h.nei_mask
        self.mp_mask = h.mp_mask

        self.mp_prob = h.mp_prob
        self.nei_rate = h.nei_rate
        self.adaptive_nei_mask = getattr(h, "adaptive_nei_mask", False)
        self.core_mp_name = set(getattr(h, "core_mp_name", []))
        self.aux_mp_name = set(getattr(h, "aux_mp_name", []))
        self.mp_stats = getattr(h, "mp_stats", {})

        self.core_mask_rate_min = getattr(h, "core_mask_rate_min", 0.10)
        self.core_mask_rate_max = getattr(h, "core_mask_rate_max", 0.20)
        self.aux_mask_rate_min = getattr(h, "aux_mask_rate_min", 0.30)
        self.aux_mask_rate_max = getattr(h, "aux_mask_rate_max", 0.50)
        self.semantic_keep_strength = getattr(h, "semantic_keep_strength", 0.20)
        self.keep_prob_floor = getattr(h, "keep_prob_floor", 0.05)
        self.keep_prob_ceiling = getattr(h, "keep_prob_ceiling", 0.98)

        self.semantic_density_range = self._build_stat_range("semantic_density")
        self.structural_sparsity_range = self._build_stat_range("structural_sparsity")

    def _build_stat_range(self, key):
        stat_values = [float(stats[key]) for stats in self.mp_stats.values()] if self.mp_stats else [0.0]
        return min(stat_values), max(stat_values)

    def _normalize_stat(self, value, stat_range):
        stat_min, stat_max = stat_range
        if abs(stat_max - stat_min) < 1e-12:
            return 0.5
        return (float(value) - stat_min) / (stat_max - stat_min)

    def _get_adaptive_mask_rate(self, mp):
        if mp not in self.mp_stats:
            return self.nei_rate

        mp_stat = self.mp_stats[mp]
        semantic_density = self._normalize_stat(mp_stat["semantic_density"], self.semantic_density_range)
        structural_sparsity = self._normalize_stat(mp_stat["structural_sparsity"], self.structural_sparsity_range)
        structural_density = 1.0 - structural_sparsity

        if mp in self.core_mp_name:
            keep_score = 0.5 * (semantic_density + structural_density)
            mask_rate = self.core_mask_rate_max - (self.core_mask_rate_max - self.core_mask_rate_min) * keep_score
        elif mp in self.aux_mp_name:
            perturb_score = 0.5 * ((1.0 - semantic_density) + structural_sparsity)
            mask_rate = self.aux_mask_rate_min + (self.aux_mask_rate_max - self.aux_mask_rate_min) * perturb_score
        else:
            mask_rate = self.nei_rate

        return float(min(max(mask_rate, 0.0), 0.95))

    def _sample_adaptive_edges(self, edge_idx, edge_weight, mp):
        edge_num = edge_idx.shape[1]
        if edge_num <= 1:
            return edge_idx

        mask_rate = self._get_adaptive_mask_rate(mp)
        base_keep_prob = 1.0 - mask_rate

        if edge_weight.numel() == 0:
            keep_prob = torch.full((edge_num,), base_keep_prob, device=edge_idx.device)
        else:
            weight_min = edge_weight.min()
            weight_max = edge_weight.max()
            if torch.abs(weight_max - weight_min) < 1e-12:
                keep_prob = torch.full((edge_num,), base_keep_prob, device=edge_idx.device)
            else:
                norm_weight = (edge_weight - weight_min) / (weight_max - weight_min)
                centered_weight = norm_weight - norm_weight.mean()
                keep_prob = base_keep_prob + self.semantic_keep_strength * centered_weight

        keep_prob = keep_prob.clamp(self.keep_prob_floor, self.keep_prob_ceiling)
        keep_mask = torch.bernoulli(keep_prob).to(dtype=torch.bool)

        if not torch.any(keep_mask):
            keep_mask[torch.argmax(keep_prob)] = True

        return edge_idx[:, keep_mask]

    def forward(self, d, full=False):
        embeds = []
        mp_list = self.mp_name[:]

        # random mask
        if self.training and self.mp_mask and not full:
            if random.random() <= self.mp_prob:
                sample_num = len(mp_list) - 1
            else:
                sample_num = len(mp_list)

            mp_list = list(np.random.choice(mp_list, sample_num, replace=False))

        for mp in mp_list:
            edge_idx = d.mp_dict[mp]._indices()
            edge_weight = d.mp_dict[mp]._values()

            if self.training and self.nei_mask and not full:
                if self.adaptive_nei_mask:
                    edge_idx = self._sample_adaptive_edges(edge_idx, edge_weight, mp)
                else:
                    edge_num = edge_idx.shape[1]
                    egde_indices = torch.randperm(edge_num)[:int(edge_num * (1 - self.nei_rate))].to(edge_idx.device)
                    edge_idx = edge_idx.index_select(1, egde_indices)
                
            attn_embed = self.node_att[mp](d.h, edge_idx)
            embeds.append(attn_embed)

        z_mp, mp_weight = self.mp_att(embeds)

        if not self.training:
            for mp, w in zip(mp_list, mp_weight):
                print("{} {:.3f}".format(mp, w), end=" ")
            print()
        return z_mp
