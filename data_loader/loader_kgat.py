import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        # CF data are obtained here
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        # getting Knowledge Graph data
        kg_data = self.load_kg(self.kg_file)
        self.__kg_statictics(kg_data)

        # constructing CKG data
        self.__construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def __kg_statictics(self, kg_data):
        self.n_relations = len(np.unique(kg_data['r']))
        self.n_entities = len(np.unique(kg_data['h'])) + len(np.unique(kg_data['t']))
        self.n_users_entities = self.n_users + self.n_entities

    def __construct_data(self, kg_data):
        '''
            1. kg_data preparation: Adding inverse kg_data to kg_data
            2. Remapping user id to make user ids unique
            3. KG_Interaction data : Adding Interaction data in train and test split to KG data
            4. Making KG_DICT : from KG_Interaction data
                a. h -> [(t, r)]
                b. r -> [(h, t)]
                c. Ordered h, r, t tensors: from KG_Interaction data rows
                    i. h tensor
                    ii. r tensor
                    iii. t tensor
        '''

        # cf_train and cf_test tuples
        self.cf_train_data = (
            self.cf_train_data[0].astype(np.int32),
            self.cf_train_data[1].astype(np.int32)
        )
        self.cf_test_data = (
            self.cf_test_data[0].astype(np.int32),
            self.cf_test_data[1].astype(np.int32)
        )

        # Train user and Test user dicts
        self.train_user_dict = {
            k: np.unique(v).astype(np.int32)
            for k, v in self.train_user_dict.items()
        }
        self.test_user_dict = {
            k: np.unique(v).astype(np.int32)
            for k, v in self.test_user_dict.items()
        }

        # creating Bi-partite graph
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        cf2kg_train_data['r'] = 442043

        # creating Collaborative Knowledge Graph (CKG)
        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct CKG data in required formats
        h_list = []; t_list = []; r_list = []
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def create_adjacency_dict(self):
        '''
            Making adjacency matrices for all relations with the heads and tails as vertical and horizontal axes
            Adjacency matrix of r1:
                    t1  t2  t3  ....
                h1  1
                h2      1
                h3          1
                .               .
                .                   .

                In the form of
                (h1, t1) 1
                (h2, t2) 1
                (h3, t3) 1
                .
                .
                .
        '''
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            ''' r, [(h, t)] '''
            rows = [e[0] for e in ht_list]  # h list
            cols = [e[1] for e in ht_list]  # t list


            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)
            vals = [1] * len(unique_rows)  # len(unique_rows or unique_cols)

            adj = sp.coo_matrix((vals, (unique_rows, unique_cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj


    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


