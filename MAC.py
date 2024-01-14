apath = r'ASTs2id.txt'
sentiment_feat_path = r'./BERT/sentiment.pkl'
dataset_name = 'Arts_Crafts_and_Sewing_5'
dataset_name_path = '../data/Arts_Crafts_and_Sewing_5/Arts_Crafts_and_Sewing_5.json'
aspect_feat_path = r'./BERT/aspect.pkl'

import argparse
import time

import dgl.function as fn
import numpy as np

from util import *
import torch
import torch.nn.functional as F
import dgl
from load_data import *
from util import *
import random
import ast
from tqdm import tqdm, trange
import json
from abc import ABC
import pickle

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)


seed_everything(2023)


class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(dataset_path)

        self.remove_users = []

        self._num_user = dataset_info["user_size"]
        self._num_item = dataset_info["item_size"]

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        self.review_feat_updated = {}
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value

        def process_sent_data(info):
            user_id = info["user_id"].to_list()
            item_id = [int(i) + self._num_user for i in info["item_id"].to_list()]
            rating = info["rating"].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self.user_item_rating = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                np.array(data[2], dtype=np.int64)

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)
            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)  ## 双向 ！！！！

            rating_values = np.concatenate([rating, rating],
                                           axis=0)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]
                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)  # 双向
        self.valid_rating_pairs, self.valid_rating_values = _generate_test_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        # generate train_review_pairs
        self.train_review_pairs = []
        for idx in range(len(self.train_rating_values)):
            u, i = self.train_rating_pairs[0][idx], self.train_rating_pairs[1][idx]
            review = self.review_feat_updated[(u, i)].numpy()
            self.train_review_pairs.append(review)

        self.train_review_pairs = np.array(self.train_review_pairs)

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph(self.train_rating_pairs, self.train_rating_values)

    def _generate_enc_graph(self, rating_pairs, rating_values):

        record_size = rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(rating_pairs[0][x], rating_pairs[1][x])] for x in
                            range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = rating_pairs
        # apath = r'X:\workspace\datasets\Amazon\@aspect\office\ASTs2id.txt'
        with open(apath, 'r') as f:
            data = f.read()
        aspect_sentiment = {}
        # tmp=[]
        for line in data.split("\n"):
            if line == "": continue
            line = line.split("####")
            id, aspect = line[0], line[1]
            if aspect == '{}': continue
            id = ast.literal_eval(id)  # [int(x) for x in id[1:-1].split(",")]
            aspect = ast.literal_eval(aspect)
            # tmp+=list(aspect.keys())
            if tuple(id) not in aspect_sentiment:
                aspect_sentiment[tuple(id)] = aspect
            else:
                for a, sjDict in aspect.items():  # {2: {2: 0, 3: 0}}
                    for s, j in sjDict.items():
                        if a in aspect_sentiment[tuple(id)]:  # {2: {2: 0, 3: 0}}
                            if s not in aspect_sentiment[tuple(id)][a]:
                                aspect_sentiment[tuple(id)][a][s] = j
                            else:
                                pass
                        else:
                            aspect_sentiment[tuple(id)][a] = sjDict
                            continue
        all_aspect = []
        for ll in [list(x.keys()) for x in list(aspect_sentiment.values())]:
            all_aspect += ll

        sentiment_feat = torch.load(sentiment_feat_path)

        num_nodes_dict = {"user": self._num_user, "item": self._num_item,
                          "aspect": len(set(all_aspect)), "review": len(self.train_datas[0])}
        # for i in  len(self.train_datas[0]):
        #     u,i,s=self.train_datas[0][i],self.train_datas[1][i],self.train_datas[2][i]
        rrow = self.train_datas[0]
        rcol = [x - self._num_user for x in self.train_datas[1]]
        graph_data = {}
        graph_data[("user", "review", "item")] = (rrow, rcol)
        graph_data[("item", "review_r", "user")] = (rcol, rrow)

        aurow = []
        aucol = []
        airow = []
        aicol = []

        aurow5 = []
        aucol5 = []
        au5s = []
        airow5 = []
        aicol5 = []
        ai5s = []

        su = []
        si = []
        ju = []
        ji = []
        sus = []
        sis = []
        a2r1 = []
        a2r2 = []
        a2rs = []
        # a2rj = []
        #############################
        for x in trange(len(rrow)):
            # if self.train_datas[2][x]!=5:continue
            if (rrow[x], rcol[x]) in aspect_sentiment:
                score = self.train_datas[2][x]
                for a, v in aspect_sentiment[(rrow[x], rcol[x])].items():
                    # a2r1 += [a]
                    # a2r2 += [x]
                    aurow += [a]
                    aucol += [rrow[x]]
                    sus += [score]
                    airow += [a]
                    aicol += [rcol[x]]
                    if True:  # score == 5:
                        for s, j in v.items():
                            # if (a,s) not in aurow5:
                            aurow5 += [(a, s)]
                            aucol5 += [rrow[x]]
                            au5s += [s]
                            airow5 += [(a, s)]
                            aicol5 += [rcol[x]]
                            ai5s += [s]
                    sis += [score]
                    a2rs_temp = []
                    su_temp = []
                    # juitemp=[]
                    for s, j in v.items():
                        a2rs_temp += [sentiment_feat[s]]
                        su_temp += [sentiment_feat[s]]
                        # juitemp += [j]
                        # ju += [j]
                        # si_temp += [sentiment_feat[s]]
                        # ji += [j]
                    a2rs_temp = torch.mean(torch.stack(a2rs_temp, 0), 0)
                    su_temp = torch.mean(torch.stack(su_temp, 0), 0)
                    # juitemp=sum(juitemp) / len(juitemp)
                    a2rs += [a2rs_temp]
                    su += [su_temp]
                    si += [su_temp]
                    # ju += [juitemp]
                    # ji += [juitemp]
            else:
                pass  # print("(rrow[x], rcol[x]) not in aspect_sentiment")

        def gen(rrow, rcol, au5s):  # user aspect
            import pickle
            u_i = {}
            i_u = {}
            u_i_s = {}
            for i in trange(len(rrow)):
                if rrow[i] not in u_i:
                    u_i[rrow[i]] = [rcol[i]]
                    # u_i_s[rrow[i]] = [rcol[i]]
                else:
                    u_i[rrow[i]] += [rcol[i]]
                if rcol[i][0] not in i_u:
                    i_u[rcol[i][0]] = [rrow[i]]
                else:
                    i_u[rcol[i][0]] += [rrow[i]]
            urow, ucol, urc = [], [], []  # u-u
            uiud = {}
            uiud_num = {}
            for u1 in tqdm(u_i.keys()):
                for u12 in u_i[u1]:
                    for u2 in i_u[u12[0]]:
                        if u1 == u2:
                            continue
                        if u1 not in uiud.keys():
                            uiud[u1] = {u2: [list(u12) + [u_i[u2][[x[0] for x in u_i[u2]].index(u12[0])][1]]]}
                            uiud_num[u1] = {u2: 1}
                        else:
                            if len(uiud[u1]) > 500:
                                break
                            if u2 not in uiud[u1].keys():
                                uiud[u1][u2] = [list(u12) + [u_i[u2][[x[0] for x in u_i[u2]].index(u12[0])][1]]]
                                uiud_num[u1][u2] = 1
                            else:
                                if u12[0] in [x[0] for x in uiud[u1][u2]]:
                                    continue
                                else:
                                    uiud[u1][u2] += [list(u12) + [u_i[u2][[x[0] for x in u_i[u2]].index(u12[0])][1]]]
                                uiud_num[u1][u2] += 1
            for u1, u2_num in uiud_num.items():
                sorted_items = sorted(u2_num.items(), key=lambda item: item[1], reverse=True)
                top_5_items = sorted_items[:25]
                sorted_dict = dict(top_5_items)
                uiud_num[u1] = sorted_dict
            for u1, u2u12 in uiud.items():
                for u2, u12s in u2u12.items():
                    if u2 not in uiud_num[u1].keys():
                        continue
                    for u12t in u12s:
                        urow += [u1]
                        ucol += [u2]
                        urc += [u12t]
            return urow, ucol, urc

        def gen2(au_a, au_u, ai_a1, ai_i):
            au_a = [x[0] for x in au_a]
            ai_a = [x[0] for x in ai_a1]
            uia = {}
            uia_n = {}
            for i in range(len(au_a)):
                if au_u[i] not in uia.keys():
                    uia[au_u[i]] = {ai_i[i]: [au_a[i]]}
                    uia_n[au_u[i]] = {ai_i[i]: 1}
                else:
                    if ai_i[i] not in uia[au_u[i]].keys():
                        uia[au_u[i]][ai_i[i]] = [au_a[i]]
                        uia_n[au_u[i]] = {ai_i[i]: 1}
                    else:
                        if au_a[i] in uia[au_u[i]][ai_i[i]]:
                            continue
                        else:
                            uia[au_u[i]][ai_i[i]] += [au_a[i]]
                            uia_n[au_u[i]][ai_i[i]] += 1
            for u1, u2_num in uia_n.items():
                sorted_items = sorted(u2_num.items(), key=lambda item: item[1], reverse=True)
                top_5_items = sorted_items[:25]
                sorted_dict = dict(top_5_items)
                uia_n[u1] = sorted_dict
            urow, ucol, urc = [], [], []
            for u1, u2u12 in uia.items():
                for u2, u12s in u2u12.items():
                    if u2 not in uia_n[u1].keys():
                        continue
                    for u12 in u12s:
                        urow += [u1]
                        ucol += [u2]
                        urc += [u12]
            return urow, ucol, urc

        re = gen(aucol5, aurow5, au5s)
        red = {}
        red[0] = re[0]
        red[1] = re[1]
        red[2] = re[2]
        import pickle
        with open('aucol-aurow-10.pkl', 'wb') as f:
            pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        #
        re = gen(aicol5, airow5, au5s)
        red = {}
        red[0] = re[0]
        red[1] = re[1]
        red[2] = re[2]
        import pickle
        with open('aicol-airow-10.pkl', 'wb') as f:
            pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        ##
        re = gen2(aurow5, aucol5, airow5, aicol5)
        red = {}
        red[0] = re[0]
        red[1] = re[1]
        red[2] = re[2]
        import pickle
        with open('aucol-aicol-10.pkl', 'wb') as f:
            pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        # print("over")
        # os.exit()
        ##<<
        with open('aucol-aurow-10.pkl', 'rb') as f:
            aucol_aurow = pickle.load(f)
        with open('aicol-airow-10.pkl', 'rb') as f:
            aicol_airow = pickle.load(f)
        with open('aucol-aicol-10.pkl', 'rb') as f:
            aucol_aicol = pickle.load(f)
        graph_data[("user", "user-aspect-user", "user")] = (aucol_aurow[0], aucol_aurow[1])
        graph_data[("item", "item-aspect-item", "item")] = (aicol_airow[0], aicol_airow[1])
        graph_data[("user", "user-aspect-item", "item")] = (aucol_aicol[0], aucol_aicol[1])
        graph_data[("item", "item-aspect-user", "user")] = (aucol_aicol[1], aucol_aicol[0])
        ##aspect高阶《《《《《《《《
        graph_data[("aspect", "aspect->user", "user")] = (aurow, aucol)
        graph_data[("aspect", "aspect->item", "item")] = (airow, aicol)

        # graph_data[("aspect", "aspect->review", "review")] = (a2r1, a2r2)

        graph = dgl.heterograph(graph_data, num_nodes_dict)
        graph.edges["review"].data["review_feat"] = review_feat_list[:int(len(review_feat_list) / 2)]

        graph.edges["review_r"].data["review_feat"] = review_feat_list[:int(len(review_feat_list) / 2)]

        graph.edges["review"].data["score"] = torch.tensor([x - 1 for x in self.train_datas[2]]).int()
        graph.edges["review_r"].data["score"] = torch.tensor([x - 1 for x in self.train_datas[2]]).int()

        graph.edges["aspect->user"].data["sentiment_feat"] = torch.stack(su, 0).float()
        graph.edges["aspect->user"].data["score"] = torch.tensor([x - 1 for x in sus]).int()
        # graph.edges["aspect->user"].data["jixing"] = torch.tensor(ju).int()
        graph.edges["aspect->item"].data["sentiment_feat"] = torch.stack(si, 0).float()
        graph.edges["aspect->item"].data["score"] = torch.tensor([x - 1 for x in sis]).int()

        # graph.edges["aspect->item"].data["jixing"] = torch.tensor(ji).int()

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        # ca_sum = _calc_norm(graph.out_degrees(etype='aspect->user'), 0.5)
        graph.nodes["user"].data["cur"] = _calc_norm(graph.out_degrees(etype='review'), 0.5)
        graph.nodes["item"].data["cir"] = _calc_norm(graph.out_degrees(etype='review_r'), 0.5)

        graph.nodes["aspect"].data["cau"] = _calc_norm(graph.out_degrees(etype='aspect->user'), 0.5)
        graph.nodes["aspect"].data["cai"] = _calc_norm(graph.out_degrees(etype='aspect->item'), 0.5)

        graph.nodes["user"].data["cau"] = _calc_norm(graph.in_degrees(etype='aspect->user'), 0.5)
        graph.nodes["item"].data["cai"] = _calc_norm(graph.in_degrees(etype='aspect->item'), 0.5)

        graph.edges["user-aspect-user"].data["aspect"] = torch.tensor(aucol_aurow[2])
        graph.edges["item-aspect-item"].data["aspect"] = torch.tensor(aicol_airow[2])
        graph.edges["user-aspect-item"].data["aspect"] = torch.tensor(aucol_aicol[2])
        graph.edges["item-aspect-user"].data["aspect"] = torch.tensor(aucol_aicol[2])
        graph.nodes["user"].data["c-user-aspect-user"] = _calc_norm(graph.in_degrees(etype="user-aspect-user"), 0.5)
        graph.nodes["user"].data["c-user-aspect-user-r"] = _calc_norm(graph.out_degrees(etype="user-aspect-user"), 0.5)
        graph.nodes["item"].data["c-item-aspect-item"] = _calc_norm(graph.in_degrees(etype="item-aspect-item"), 0.5)
        graph.nodes["item"].data["c-item-aspect-item-r"] = _calc_norm(graph.out_degrees(etype="item-aspect-item"), 0.5)
        graph.nodes["user"].data["c-user-aspect-item"] = _calc_norm(graph.out_degrees(etype="user-aspect-item"), 0.5)
        graph.nodes["item"].data["c-user-aspect-item"] = _calc_norm(graph.in_degrees(etype="user-aspect-item"), 0.5)
        graph.nodes["item"].data["c-item-aspect-user"] = _calc_norm(graph.out_degrees(etype="item-aspect-user"),
                                                                    0.5)
        graph.nodes["user"].data["c-item-aspect-user"] = _calc_norm(graph.in_degrees(etype="item-aspect-user"),
                                                                    0.5)
        return graph

    def _train_data(self, batch_size=1024):
        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        np.random.shuffle(idx)
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1 if data_len != batch_size else 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]
            batch_reviews = self.train_review_pairs[begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

            review_list.append(torch.FloatTensor(batch_reviews).to('cuda:0'))

        return u_list, i_list, r_list

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list


def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.model_short_name = 'RGC'
    args.dataset_name = dataset_name
    args.dataset_path = dataset_name_path
    args.emb_size = 64
    args.emb_dim = 64

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    args.batch_size = 271466

    return args


gloabl_dropout = 0.7


class GCN(nn.Module):
    def __init__(self, params, dropout_rate):
        super(GCN, self).__init__()
        self.num_users = params.num_users
        self.num_items = params.num_items
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(0.4)
        self.score = nn.Embedding(5, params.emb_dim * 4)
        self.score_r = nn.Embedding(5, params.emb_dim * 4)
        # self.score_a = nn.Embedding(5, params.emb_dim*3)
        # self.score_a_r = nn.Embedding(5, params.emb_dim*3)

        self.review_w = nn.Linear(params.emb_size, params.emb_dim, bias=False)
        self.review_r_w = nn.Linear(params.emb_size, params.emb_dim, bias=False)
        # self.sentiment_a_r = nn.Linear(params.emb_size, params.emb_dim, bias=False)  ###
        self.aspect_w = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.aspect_w_r = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.aspect_w_3 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.sentiment_w = nn.Linear(params.emb_dim, params.emb_dim, bias=False)  ###
        self.sentiment_w_r = nn.Linear(params.emb_dim, params.emb_dim, bias=False)

        self.aspect_feat = torch.stack(list(torch.load(aspect_feat_path).
                                            values())).to(torch.float32).cuda()
        self.sentiment_feat = torch.stack(list(torch.load(sentiment_feat_path).
                                               values())).to(torch.float32).cuda()
        self.s1 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.s2 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.weight = nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        # self.gru1=nn.GRU(params.emb_dim,params.emb_dim)
        # self.gru2 = nn.GRU(params.emb_dim, params.emb_dim)

    def forward(self, g, feature):
        g.nodes["user"].data["fe"], g.nodes["item"].data["fe"] = torch.split(feature, [self.num_users, self.num_items],
                                                                             dim=0)
        g.nodes["user"].data["fee"], g.nodes["item"].data["fee"] = torch.split(self.weight,
                                                                               [self.num_users, self.num_items],
                                                                               dim=0)
        g.nodes["aspect"].data["fe"] = self.aspect_w(self.aspect_feat)
        g.nodes["aspect"].data["fe1"] = self.aspect_w_r(self.aspect_feat)

        g.edges["review"].data["r"] = self.review_w(g.edges["review"].data["review_feat"])
        g.edges["review_r"].data["r"] = self.review_r_w(g.edges["review_r"].data["review_feat"])
        g.edges["review"].data["s"] = self.score(g.edges["review"].data["score"])
        g.edges["review_r"].data["s"] = self.score_r(g.edges["review_r"].data["score"])

        g.edges["aspect->user"].data["r"] = self.sentiment_w(g.edges["aspect->user"].data["sentiment_feat"])
        # g.edges["aspect->user"].data["a_score"] = self.score_a(g.edges["aspect->user"].data["score"])
        g.edges["aspect->item"].data["r"] = self.sentiment_w_r(g.edges["aspect->item"].data["sentiment_feat"])
        # g.edges["aspect->item"].data["a_score"] = self.score_a_r(g.edges["aspect->item"].data["score"])
        # g.edges["aspect->review"].data["r"] = self.sentiment_w_3(g.edges["aspect->review"].data["sentiment_feat"])

        # g.edges["user<item>user"].data["r"] = g.nodes["item"].data["fe"][g.edges["user<item>user"].data["item"]]
        # g.edges["item<user>item"].data["r"] = g.nodes["user"].data["fe"][g.edges["item<user>item"].data["user"]]
        g.edges["user-aspect-user"].data["r"] = g.nodes["aspect"].data["fe1"][
            g.edges["user-aspect-user"].data["aspect"][:, 0]]
        g.edges["item-aspect-item"].data["r"] = g.nodes["aspect"].data["fe1"][
            g.edges["item-aspect-item"].data["aspect"][:, 0]]
        g.edges["user-aspect-item"].data["r"] = g.nodes["aspect"].data["fe1"][
            g.edges["user-aspect-item"].data["aspect"]]
        g.edges["item-aspect-user"].data["r"] = g.nodes["aspect"].data["fe1"][
            g.edges["item-aspect-user"].data["aspect"]]

        s1 = self.s1(self.sentiment_feat)
        s2 = self.s2(self.sentiment_feat)
        g.edges["user-aspect-user"].data["s1"] = s1[g.edges["user-aspect-user"].data["aspect"][:, 1]]
        g.edges["item-aspect-item"].data["s1"] = s2[g.edges["item-aspect-item"].data["aspect"][:, 1]]
        g.edges["user-aspect-user"].data["s2"] = s1[g.edges["user-aspect-user"].data["aspect"][:, 2]]
        g.edges["item-aspect-item"].data["s2"] = s2[g.edges["item-aspect-item"].data["aspect"][:, 2]]

        funcs = {
            "aspect->user": (lambda edges: {
                'm': ((edges.src["fe"] + edges.data["r"])) * self.dropout1(
                    edges.src["cau"])}, fn.sum(msg='m', out='h')),
            "aspect->item": (lambda edges: {
                'm': ((edges.src["fe"] + edges.data["r"])) * self.dropout1(
                    edges.src["cai"])}, fn.sum(msg='m', out='h')),
            "user-aspect-user": (
                lambda edges: {
                    'm': (edges.src["fe"] + edges.data["r"]) * torch.sigmoid(
                        edges.data["s1"] + edges.data["s2"]) * self.dropout(edges.src["c-user-aspect-user"])},
                fn.sum(msg='m', out='h1')),
            "item-aspect-item": (
                lambda edges: {
                    'm': (edges.src["fe"] + edges.data["r"]) * torch.sigmoid(
                        edges.data["s1"] + edges.data["s2"]) * self.dropout(edges.src["c-item-aspect-item"])},
                fn.sum(msg='m', out='h2')),
            "user-aspect-item": (
                lambda edges: {
                    'm': (edges.src["fee"] + edges.data["r"]) * self.dropout(edges.src["c-user-aspect-item"])},
                fn.sum(msg='m', out='h3')),
            "item-aspect-user": (
                lambda edges: {
                    'm': (edges.src["fee"] + edges.data["r"]) * self.dropout(edges.src["c-item-aspect-user"])},
                fn.sum(msg='m', out='h3')),
        }
        g.multi_update_all(funcs, "stack")
        g.nodes["user"].data["from_a"] = torch.cat([g.nodes["user"].data["h"][:, 0, :] * g.nodes["user"].data["cau"], \
                                                    g.nodes["user"].data["h1"][:, 0, :] * g.nodes["user"].data[
                                                        "c-user-aspect-user-r"], \
                                                    g.nodes["user"].data["h3"][:, 0, :] * g.nodes["user"].data[
                                                        "c-user-aspect-item"],
                                                    ], -1)
        g.nodes["item"].data["from_a"] = torch.cat([g.nodes["item"].data["h"][:, 0, :] * g.nodes["item"].data["cai"], \
                                                    g.nodes["item"].data["h2"][:, 0, :] * g.nodes["item"].data[
                                                        "c-item-aspect-item-r"], \
                                                    g.nodes["item"].data["h3"][:, 0, :] * g.nodes["item"].data[
                                                        "c-item-aspect-user"],
                                                    ], -1)

        funcs1 = {
            "review": (lambda edges: {'m': (torch.cat([edges.src["from_a"], edges.data["r"]], -1)) * torch.sigmoid(
                edges.data["s"]) * self.dropout(edges.src["cur"])}, fn.sum(msg='m', out='h')),
            "review_r": (lambda edges: {'m': (torch.cat([edges.src["from_a"], edges.data["r"]], -1)) * torch.sigmoid(
                edges.data["s"]) * self.dropout(edges.src["cir"])}, fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs1, "stack")
        g.nodes["user"].data["fe"] = g.nodes["user"].data["h"][:, 0, :] * g.nodes["user"].data["cur"]
        g.nodes["item"].data["fe"] = g.nodes["item"].data["h"][:, 0, :] * g.nodes["item"].data["cir"]

        return torch.cat([g.nodes["user"].data["fe"], g.nodes["item"].data["fe"]], 0), 0  # ,#(l1+l2)/2


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        print("#################", params)
        self.weight = nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))

        self.encoder = GCN(params, gloabl_dropout)

        self.num_user = params.num_users
        self.num_item = params.num_items

        self.fc_user2 = nn.Linear(params.emb_dim * 4, params.emb_dim * 4)
        self.fc_item2 = nn.Linear(params.emb_dim * 4, params.emb_dim * 4)

        self.dropout1 = nn.Dropout(0.4)  # 0.3)#gloa-bl_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.predictor1 = nn.Sequential(
            nn.Linear(params.emb_dim * 4, params.emb_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(params.emb_dim * 4, 5, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_graph_dict, users, items):

        feat, l12 = self.encoder(enc_graph_dict, self.weight)

        u_feat, i_feat = torch.split(feat, [self.num_user, self.num_item], dim=0)

        ua = self.fc_user2(self.dropout1(u_feat))
        ia = self.fc_item2(self.dropout1(i_feat))

        feat = torch.cat([ua, ia], dim=0)
        user_embeddings, item_embeddings = feat[users], feat[items]
        pred_ratings2 = self.predictor1(user_embeddings * item_embeddings)
        # l12=(self.contrast_loss(ur0,ua)+self.contrast_loss(ir0,ia))/2
        return pred_ratings2, 0  # l12#0  # +pred_ratings2


def evaluate(args, net, dataset, flag='valid'):
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    enc_graph = dataset.train_enc_graph
    # graph_aspect=dataset.graph_aspect

    net.eval()
    with torch.no_grad():
        pred_ratings, _ = net(enc_graph, u_list, i_list)

        real_pred_ratings = (torch.softmax(pred_ratings, dim=1) *
                             nd_possible_rating_values.view(1, -1)).sum(dim=1)

        u_list = u_list.cpu().numpy()
        r_list = r_list.cpu().numpy()
        real_pred_ratings = real_pred_ratings.cpu().numpy()

        mse = ((real_pred_ratings - r_list) ** 2.).mean()

    return mse


def train(params):
    dataset = Data(params.dataset_name,
                   params.dataset_path,
                   params.device,
                   params.emb_size,
                   )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values

    print(
        f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_test_mse = np.inf
    no_better_valid = 0
    best_iter = -1

    # for r in [1, 2, 3, 4, 5]:
    #     dataset.train_enc_graph[str(r)] = dataset.train_enc_graph[str(r)].int().to(params.device)
    dataset.train_enc_graph = dataset.train_enc_graph.int().to(
        params.device)
    # dataset.graph_aspect=dataset.graph_aspect.int().to(params.device)

    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(params.device)

    print("Training and evaluation.")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        # n_batch = len(dataset.train_rating_values) // params.batch_size + 1
        u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
        train_mse = 0.

        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]
            pred_ratings, l12 = net(dataset.train_enc_graph, batch_user, batch_item)

            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

            loss = rating_loss_net(pred_ratings, batch_rating).mean() + l12  ##########!!!!!!!!!!!!

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_mse += ((real_pred_ratings - batch_rating - 1) ** 2).sum()

        train_mse = train_mse / len(dataset.train_rating_values)

        # valid_mse = evaluate(args=params, net=net, dataset=dataset, flag='valid')

        test_mse = evaluate(args=params, net=net, dataset=dataset, flag='test')

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience:
                print("Early stopping threshold reached. Stop training.")
                break

        print(
            f'Epoch {iter_idx}, Loss={loss:.4f}, Train_MSE={train_mse:.4f}, Valid_MSE={0:.4f}, Test_MSE={test_mse:.4f}')
    import datetime

    current_time = datetime.datetime.now()
    print(current_time)
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_mse:.4f}')


if __name__ == '__main__':
    config_args = config()
    train(config_args)
