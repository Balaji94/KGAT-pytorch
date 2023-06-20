import os
import sys
import random
from flask import Flask, jsonify
from time import time

import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT

app = Flask("KGAT")

class KGAT_wrapper:

    data = None

    def __init__(self):
        # seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # initialize logging
        log_save_id = create_log_id(args.save_dir)
        logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
        logging.info(args)

        # GPU / CPU config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __train_cf_batch(self, batch, data, model, cf_optimizer):
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict,
                                                                                     data.cf_batch_size)
        cf_batch_user = cf_batch_user.to(self.device)
        cf_batch_pos_item = cf_batch_pos_item.to(self.device)
        cf_batch_neg_item = cf_batch_neg_item.to(self.device)

        cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

        if np.isnan(cf_batch_loss.cpu().detach().numpy()):
            logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, batch, n_cf_batch))
            sys.exit()

        cf_batch_loss.backward()
        cf_optimizer.step()
        cf_optimizer.zero_grad()

        return cf_batch_loss

    def __train_kg_batch(self, batch, data, model, kg_optimizer):
        kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
        kg_batch_head = kg_batch_head.to(self.device)
        kg_batch_relation = kg_batch_relation.to(self.device)
        kg_batch_pos_tail = kg_batch_pos_tail.to(self.device)
        kg_batch_neg_tail = kg_batch_neg_tail.to(self.device)

        kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

        if np.isnan(kg_batch_loss.cpu().detach().numpy()):
            logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, batch, n_kg_batch))
            sys.exit()

        kg_batch_loss.backward()
        kg_optimizer.step()
        kg_optimizer.zero_grad()

        return kg_batch_loss

    def __update_attention(self, data, model):
        h_list = data.h_list.to(self.device)
        t_list = data.t_list.to(self.device)
        r_list = data.r_list.to(self.device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')

    def train(self, args):
        print("Training Started...")

        data = self.data

        if args.use_pretrain == 1:
            user_pre_embed = torch.tensor(data.user_pre_embed)
            item_pre_embed = torch.tensor(data.item_pre_embed)
        else:
            user_pre_embed, item_pre_embed = None, None

        # construct model
        print("Constructing Model...")
        model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
        if args.use_pretrain == 2:model = load_model(model, args.pretrain_model_path)
        model.to(self.device)
        logging.info(model)

        # construct optimizer
        cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # initialize metrics
        best_epoch = -1
        best_recall = 0
        Ks = eval(args.Ks)
        k_min = min(Ks)
        k_max = max(Ks)
        epoch_list = []
        metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

        # train model
        for epoch in range(1, args.n_epoch + 1):
            print(f"Model epoch {epoch}")
            time0 = time()
            model.train()

            # train cf
            time1 = time()
            cf_total_loss = 0
            n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
            for iter in range(1, n_cf_batch + 1):
                time2 = time()
                cf_batch_loss = self.__train_cf_batch(iter, data, model, cf_optimizer)
                cf_total_loss += cf_batch_loss.item()
                if (iter % args.cf_print_every) == 0:
                    logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
            logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

            # train kg
            time3 = time()
            kg_total_loss = 0
            n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
            for iter in range(1, n_kg_batch + 1):
                time4 = time()
                kg_batch_loss = self.__train_kg_batch(iter, data, model, kg_optimizer)
                kg_total_loss += kg_batch_loss.item()
                if (iter % args.kg_print_every) == 0:
                    logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
            logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

            # update attention
            time5 = time()
            self.__update_attention(data, model)
            logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

            logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            # evaluate cf
            if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
                time6 = time()

                _, metrics_dict = self.evaluate(model, data, Ks, self.device)

                logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

                epoch_list.append(epoch)
                for k in Ks:
                    for m in ['precision', 'recall', 'ndcg']:
                        metrics_list[k][m].append(metrics_dict[k][m])
                best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

                if should_stop:
                    print("Early stopping...")
                    break

                if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                    print("Saving model...")
                    save_model(model, args.save_dir, epoch, best_epoch, final_path=args.pretrain_model_path)
                    logging.info('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch


        # save metrics
        metrics_df = [epoch_list]
        metrics_cols = ['epoch_idx']
        for k in Ks:
            for m in ['precision', 'recall', 'ndcg']:
                metrics_df.append(metrics_list[k][m])
                metrics_cols.append('{}@{}'.format(m, k))
        metrics_df = pd.DataFrame(metrics_df).transpose()
        metrics_df.columns = metrics_cols
        metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

        # print best metrics
        best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))

        self.data = data

        print("Training Completed successfully...")

    def evaluate(self, model, dataloader, Ks, device, test_user_ids=None):
        model.eval()

        test_batch_size = dataloader.test_batch_size
        train_user_dict = dataloader.train_user_dict
        test_user_dict = dataloader.test_user_dict

        user_ids = list(test_user_dict.keys()) if test_user_ids is None else [dataloader.remap_id(og_id) for og_id in test_user_ids]
        user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

        item_ids = torch.LongTensor([dataloader.remap_id(id) for id in dataloader.items])

        cf_scores = []
        metric_names = ['precision', 'recall', 'ndcg']
        metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

        with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
            for batch_user_ids in user_ids_batches:
                batch_user_ids = batch_user_ids.to(self.device)

                with torch.no_grad():
                    batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

                batch_scores = batch_scores.cpu()
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

                cf_scores.append(batch_scores.numpy())
                for k in Ks:
                    for m in metric_names:
                        metrics_dict[k][m].append(batch_metrics[k][m])
                pbar.update(1)

        cf_scores = np.concatenate(cf_scores, axis=0)
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
        return cf_scores, metrics_dict

    def predict(self, args):
        # GPU / CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load data
        # data_loader_processed = os.path.join(args['data_dir'], 'pretrain', args['data_name'], 'data_loader', 'dataloader.pkl')

        if self.data is None:
            self.data = DataLoaderKGAT(args, logging)
        data = self.data

        # load model
        model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
        model = load_model(model, args.pretrain_model_path)
        model.to(device)

        # predict
        Ks = eval(args.Ks)
        k_min = min(Ks)
        k_max = max(Ks)
        cf_scores, metrics_dict = self.evaluate(model, data, Ks, self.device)

        np.save(args.save_dir + 'cf_scores.npy', cf_scores)
        print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
        self.data = data

        return cf_scores, metrics_dict

    def recommend_candidates(self, job_id, k=5):
        return 0

    def recommend_jobs(self, candidate_id, k=5):
        return 1

args = None
kgat_wrapper = None

@app.route("/")
def home():
    return "KGAT POC"

@app.route("/train", methods=['GET'])
def train():
    kgat_wrapper.train(args)
    return "Training Done!!!"

@app.route("/predict", methods=['GET'])
def predict():
    cf_scores, metrics_dict = kgat_wrapper.predict(args)
    return str(cf_scores)

@app.route("/jobs", methods=['GET'])
def get_jobs():
    return str(kgat_wrapper.data.users)

@app.route("/candidates", methods=['GET'])
def get_candidates():
    return str(kgat_wrapper.data.items)

@app.route("/rc/<jobid>", methods=['GET'])
def rc(jobid):
    top_k = 10
    top_k_recommendations = kgat_wrapper.recommend_candidates(job_id, top_k)
    return top_k_recommendations

@app.route("/rj/<candidateid>", methods=['GET'])
def rj(candidateid):
    top_k = 10
    top_k_recommendations = kgat_wrapper.recommend_jobs(candidate_id, top_k)
    return top_k_recommendations



if __name__ == '__main__':

    args = parse_kgat_args()
    args.pretrain_model_path = args.pretrain_model_path.replace("model.pth", "kgat_model_recruit.pth")

    kgat_wrapper = KGAT_wrapper()
    kgat_wrapper.data = DataLoaderKGAT(args, logging)

    app.run(port=8000, debug=True)




