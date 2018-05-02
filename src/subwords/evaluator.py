from collections import defaultdict
import util
import torch
from torch.autograd import Variable
import numpy as np
import json
import platform
import time
from sklearn.utils.extmath import randomized_svd
import codecs
import scipy
from os import path

from util import read_validation_file
from util import drop_oov_from_validation_set
from util import pad

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys

VOCABSIZE = 10000

class Evaluator:
    def __init__(self, params, src_data, tgt_data):
        self.params = params
        self.batch_size = params.mini_batch_size
        self.data_dir = params.data_dir
        self.top_ks = params.top_ks
        self.sim_metrics = params.sim_metrics
        # self.models = params.models
        # self.refine = params.refine
        self.csls_k = params.csls_k
        # self.mask_procrustes = params.mask_procrustes
        # self.num_refine = params.num_refine

        self.src_n_vocab = src_data['E'].size()[0]
        self.src_indexer = src_data['id2idx']
        self.tgt_n_vocab = tgt_data['E'].size()[0]
        self.tgt_indexer = tgt_data['word2idx']

        self.valid = []
        print('Eval: {}'.format(params.validation_file))
        self.valid.append(self.setup_validation_data(params.validation_file))

        self.cosine_top = params.cosine_top
        self.refine_top = params.refine_top

    def setup_validation_data(self, filename):
        """Read a validation set from a file."""
        src_indices, src_seqs, tgt_indices = read_validation_file(
            path.join(self.data_dir, filename), self.src_indexer, self.tgt_indexer)
        # Drop pairs that contain OOV words (subwords)
        src_seqs, tgt_indices = drop_oov_from_validation_set(
            src_seqs, tgt_indices, self.src_n_vocab, self.tgt_n_vocab)

        valid = {}
        valid['valid_src_subword_ids'] = torch.LongTensor(pad(src_seqs))
        # if not self.params.disable_cuda and torch.cuda.is_available():  # TODO
        #     valid['valid_src_subword_ids'] = valid['valid_src_subword_ids'].cuda()

        valid['valid_src_word_ids'] = src_indices
        valid['valid_tgt_word_ids'] = tgt_indices
        valid['gold'] = defaultdict(list)
        for src, tgt in zip(src_indices, tgt_indices):
            valid['gold'][src].append(tgt)

        return valid

    def precision(self, g, src_data, tgt_data):
        """Evaluate precision."""
        # # Fetch embeddings
        # batches = src_data['seqs'][:VOCABSIZE].split(self.batch_size)
        # mapped_src_emb = torch.cat([g(src_data['F'](batch, src_data['E'])).detach()
        #                             for batch in batches])

        # # Normalize the embeddings
        # mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
        # tgt_emb = tgt_emb / tgt_emb.norm(2, 1)[:, None]

        # # Calculate r_target
        # if 'csls' in self.sim_metrics:
        #     print("Calculating r_target...")
        #     start_time = time.time()
        #     self.r_target = common_csls_step(self.csls_k, mapped_src_emb, tgt_emb)
        #     print("Time taken for making r_target: ", time.time() - start_time)


        # if 'with-ref' in self.refine:
        #     print("Performing refinement...")
        #     start_time = time.time()
        #     for _ in range(self.num_refine):
        #         mapped_src_emb = self.get_refined_mapping(mapped_src_emb, tgt_emb)
        #         mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
        #         self.r_target = common_csls_step(self.csls_k, mapped_src_emb, tgt_emb)
        #     refined_mapped_src_emb = mapped_src_emb
        #     print("Time taken for refinement: ", time.time() - start_time)

        # adv_mapped_src_emb = mapped_src_emb

        # Target-side embeddings
        tgt_emb = tgt_data['E'].emb.weight
        tgt_emb /= tgt_emb.norm(2, dim=1).view((-1, 1))

        start_time = time.time()
        all_precisions = {}

        buckets = None
        save = False

        for it, v in enumerate(self.valid):
            dataset = 'validation' if it == 0 else 'validation-new'
            all_precisions[dataset] = {}

            # Fetch embeddings
            batches = v['valid_src_subword_ids'].split(self.batch_size)
            if g.map1.weight.is_cuda:
                mapped_src_emb = torch.cat([g(src_data['F'](batch, src_data['E']).cuda()).detach()
                                            for batch in batches])
            else:
                mapped_src_emb = torch.cat([g(src_data['F'](batch, src_data['E'])).detach()
                                            for batch in batches])
            mapped_src_emb /= mapped_src_emb.norm(2, dim=1).view((-1, 1))
            for m in self.sim_metrics:
                indices = self.__get_knn(tgt_emb, mapped_src_emb,
                                         k=max(self.top_ks), metric=m)
                all_precisions[dataset][m] = {}
                for k in self.top_ks:
                    p = self.__precision(
                        pred=indices[:, :k], src_indices=v['valid_src_word_ids'],
                        gold=v['gold'], buckets=buckets, save=save)
                    all_precisions[dataset][m][k] = p
            # for mod in self.models:
            #     if mod == 'procrustes':
            #         mapped_src_emb = procrustes_mapped_src_emb.clone()
            #     elif mod == 'adv':
            #         mapped_src_emb = adv_mapped_src_emb.clone()
            #     else:
            #         raise 'Model not implemented: %s' % mod
                    
            #     all_precisions[key][mod] = {}

            #     for r in self.refine:
            #         if r == 'with-ref':
            #             if mod == 'procrustes':
            #                 continue
            #             else:
            #                 mapped_src_emb = refined_mapped_src_emb.clone()
                    
            #         mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
                    
            #         if 'csls' in self.methods:
            #             self.r_source = common_csls_step(
            #                 self.csls_k, tgt_emb, mapped_src_emb[v['valid_src_word_ids']])
            #             start_time = time.time()
            #             self.r_target = common_csls_step(
            #                 self.csls_k, mapped_src_emb, tgt_emb)

            #         all_precisions[key][mod][r] = {}

            #         for m in self.methods:
            #             all_precisions[key][mod][r][m] = {}

            #             for k in self.ks:
            #                 if key == 'validation-new' and mod == 'adv' and r == 'with-ref' and m == 'csls' and k == 1:
            #                     buckets = 5
            #                     save = True

            #                 p = self.get_precision_k(k, tgt_emb, mapped_src_emb, v, method=m, buckets=buckets, save=save)
            #                 if not save:
            #                     print("key: %s, model: %s, refine: %s, method: %s, k: %d, prec: %f" % (key, mod, r, m, k, p))
            #                 else:
            #                     print("key: %s, model: %s, refine: %s, method: %s, k: %d" % (key, mod, r, m, k))
            #                     print("precision: ", p)
            #                 all_precisions[key][mod][r][m][k] = p

            #                 buckets = None
            #                 save = False
        print('Time taken to run main loop: ', time.time() - start_time)
        print(json.dumps(all_precisions, indent=2))
        return all_precisions

    def __get_knn(self, tgt_emb, mapped_src_emb, k=1, metric='csls'):
        if metric == 'nn':
            _, indices = get_knn_indices(k, tgt_emb, mapped_src_emb)
            return indices

        if metric == 'csls':
            raise NotImplementedError('CSLS is not implemented yet')
            r_source = calc_csls_discount(
                self.csls_k, mapped_src_emb, mapped_src_emb)
            r_source = calc_csls_discount(
                self.csls_k, tgt_emb, mapped_src_emb)
            indices = get_csls_indices(k, xb, xq, self.r_source, self.r_target)
            return indices

        raise NotImplementedError('Metric not implemented: ' + metric)

    def __precision(self, pred, src_indices, gold, buckets=None, save=False):
        hits = {i: 0 for i in gold.keys()}  # correct or not
        # TODO: dict
        for src_idx, nn in zip(src_indices, pred):
            if len(set(gold[src_idx]).intersection(nn)) > 0:
                hits[src_idx] = 1

        return sum(hits.values()) / float(len(hits))

    def calc_unsupervised_criterion(self, mapped_src_emb):
        src_wrd_ids = torch.arange(self.cosine_top).type(torch.LongTensor)
        start_time = time.time()
        xq = mapped_src_emb[src_wrd_ids]
        xb = self.tgt_emb / self.tgt_emb.norm(2, 1)[:, None]
        r_source = common_csls_step(1, xb, xq)

        if self.r_target is None:
            print("Calculating r_target...")
            start_time = time.time()
            self.r_target = common_csls_step(self.csls_k, mapped_src_emb, xb)
            print("Time taken for making r_target: ", time.time() - start_time)

        knn_indices = csls(1, xb, xq, r_source, self.r_target)
        knn_indices = knn_indices.view(knn_indices.numel())
        sim = xq.mm(self.tgt_emb[knn_indices].transpose(0, 1))
        print(sim.mean())
        print("Time taken for computation of unsupervised criterion: ", time.time()-start_time)
        return sim.mean()

    def get_precision_k(self, k, tgt_emb, mapped_src_emb, v,
                        method='csls', buckets=None, save=False):
        n = 1500
        tgt_word_ids = v['valid_tgt_word_ids']

        if method == 'nn':
            _, knn_indices = get_knn_indices(
                k, tgt_emb=tgt_emb, src_emb=mapped_src_emb)
        elif method == 'csls':
            knn_indices = csls(k, xb, xq, self.r_source, self.r_target)

        else:
            raise "Method not implemented: %s" % method

        p, c = _calculate_precision(n, knn_indices, tgt_word_ids, buckets)
        # if save:
        #     _save_learnt_dictionary(self.data_dir, v, self.tgt_id2wrd, knn_indices, c)

        return p

    def get_refined_mapping(self, mapped_src_emb, tgt_emb):
        pairs = self.learn_refined_dictionary(mapped_src_emb, tgt_emb)
        return self.do_procrustes(pairs)


    def learn_refined_dictionary(self, mapped_src_emb, tgt_emb):
        bs = 15000
        pairs = None
        for i in range(0, self.refine_top, bs):
            lo = i
            hi = min(self.refine_top, i + bs)
            src_wrd_ids = torch.arange(lo, hi).type(torch.LongTensor)
            top_src_emb = mapped_src_emb[src_wrd_ids]
            r_source = common_csls_step(self.csls_k, tgt_emb, top_src_emb)
            knn_indices = csls(1, tgt_emb, top_src_emb, r_source, self.r_target)
            temp = torch.cat([src_wrd_ids[:, None], knn_indices], 1)
            if pairs is None:
                pairs = temp
            else:
                pairs = torch.cat([pairs, temp], 0)
        pairs = _mask(pairs, self.refine_top)
        # src_wrd_ids = torch.arange(self.refine_top).type(torch.LongTensor)
        # top_src_emb = mapped_src_emb[src_wrd_ids]
        # r_source = common_csls_step(self.csls_k, tgt_emb, top_src_emb)
        # knn_indices = csls(1, tgt_emb, top_src_emb, r_source, self.r_target)
        # pairs = torch.cat([src_wrd_ids[:, None], knn_indices], 1)
        # pairs = _mask(pairs, self.refine_top)
        return pairs

    def do_procrustes(self, pairs):
        W =_procrustes(pairs, self.src_emb, self.tgt_emb)
        mapped_src_emb = W.mm(self.src_emb.transpose(0, 1)).transpose(0, 1)
        return mapped_src_emb

    def process_gold_file(self):
        gold_dict_ids = util.map_dict2ids(self.data_dir, self.gold_file, self.suffix_str)
        gold_src_ids = []
        gold_tgt_ids = []
        for src, tgt_list in gold_dict_ids.items():
            for tgt in tgt_list:
                gold_src_ids.append(src)
                gold_tgt_ids.append(tgt)
        gold_src_tensor = util.to_tensor(np.array(gold_src_ids)).type(torch.LongTensor)
        gold_tgt_tensor = util.to_tensor(np.array(gold_tgt_ids)).type(torch.LongTensor)
        pairs = torch.cat([gold_src_tensor[:, None], gold_tgt_tensor[:, None]], 1)
        if self.mask_procrustes:
            pairs = _mask(pairs, self.refine_top)
        pairs.type(torch.LongTensor)
        return pairs


def _mask(pairs, thresh):
    mask = pairs[:, 1] <= thresh
    mask = torch.cat([mask[:, None], mask[:, None]], 1)
    selected_pairs = torch.masked_select(pairs, mask).view(-1, 2)
    return selected_pairs


def get_knn_indices(k, tgt_emb, src_emb):
    tgt_emb = (tgt_emb.cpu() if tgt_emb.is_cuda else tgt_emb).numpy()
    src_emb = (src_emb.cpu() if src_emb.is_cuda else src_emb).numpy()
    d = src_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    # if hasattr(faiss, 'StandardGpuResources'):
    #     res = faiss.StandardGpuResources()
    #     config = faiss.GpuIndexFlatConfig()
    #     config.device = 0
    #     index = faiss.GpuIndexFlatIP(res, d, config)
    # else:
    #     index = faiss.IndexFlatIP(d)

    index.add(tgt_emb)
    distances, knn_indices = index.search(src_emb, k)
    return distances, knn_indices


def _procrustes(pairs, src_emb, tgt_emb):
    X = np.transpose(src_emb[pairs[:, 0]].cpu().numpy())
    Y = np.transpose(tgt_emb[pairs[:, 1]].cpu().numpy())
#     U, Sigma, VT = randomized_svd(np.matmul(Y, np.transpose(X)), n_components=np.shape(X)[0])
    U, Sigma, VT = scipy.linalg.svd(np.matmul(Y, np.transpose(X)), full_matrices=True)
    W = util.to_tensor(np.matmul(U, VT)).type(torch.FloatTensor)
    return W


def _calculate_precision(n, knn_indices, tgt_word_ids, buckets=None):
    p, c = _calc_prec(n, knn_indices, tgt_word_ids)
    if buckets is not None:
        width = int(n/buckets)
        prec_list = [round(p, 2)]
        for i in range(0, n, width):
            lo = i
            hi = i + width
            p, _ = _calc_prec(width, knn_indices[lo:hi], tgt_word_ids, lo)
            prec_list.append(round(p, 2))
        return prec_list, c
    else:
        return round(p, 2), c


def _calc_prec(n, knn_indices, tgt_word_ids, lo=0):
    p = 0.0
    c = []
    for i, knn in enumerate(knn_indices):
        if len(set(knn).intersection(set(tgt_word_ids[i + lo]))) > 0:
            p += 1.0
            c.append(1)
        else:
            c.append(0)
    return (p/n)*100, c


def calc_csls_discount(k, tgt_emb, mapped_src_emb):
    distances, _ = get_knn_indices(k, tgt_emb, mapped_src_emb)
    return torch.from_numpy(np.average(distances, axis=1)).float()


def get_csls_indices(k, xb, xq, r_source, r_target):
    csls = 2 * xq.mm(xb.transpose(0, 1))
    csls.sub_(r_source[:, None] + r_target[None, :])
    knn_indices = csls.topk(k, dim=1)[1]
    return knn_indices


def _save_learnt_dictionary(data_dir, v, tgt_id2wrd, knn_indices, correct_or_not):
    true_dict = v['valid_dict']
    src_wrd_ids = v['valid_src_word_ids']
    src_wrds = list(true_dict.keys())

    learnt_dict_correct = {}
    learnt_dict_incorrect = {}
    bucket_list_correct = {}
    bucket_list_incorrect = {}
    src_wrd_ids_correct = {}
    src_wrd_ids_incorrect = {}

    for i, w in enumerate(src_wrds):
        bucket = int(i/300)
        if correct_or_not[i] == 0:
            learnt_dict = learnt_dict_incorrect
            bucket_list_incorrect[w] = bucket
            src_wrd_ids_incorrect[w] = src_wrd_ids[i]
        else:
            learnt_dict = learnt_dict_correct
            bucket_list_correct[w] = bucket
            src_wrd_ids_correct[w] = src_wrd_ids[i]
        learnt_dict[w] = {}
        pass
        learnt_dict[w]['true'] = true_dict[w]
        learnt_dict[w]['predicted'] = []
        for knn in knn_indices[i]:
            learnt_dict[w]['predicted'].append(tgt_id2wrd[knn])

    with codecs.open(data_dir + 'correct.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(learnt_dict_correct, ensure_ascii=False, indent=2))

    with codecs.open(data_dir + 'incorrect.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(learnt_dict_incorrect, ensure_ascii=False, indent=2))

    _write_csv(src_wrd_ids_correct, bucket_list_correct, data_dir, 'correct.csv', learnt_dict_correct)
    _write_csv(src_wrd_ids_incorrect, bucket_list_incorrect, data_dir, 'incorrect.csv', learnt_dict_incorrect)


def _write_csv(src_wrd_ids, bucket_list, data_dir, fname, learnt_dict):
    with codecs.open(data_dir + fname, 'w', encoding='utf-8') as f:
        f.write("Bucket, Word ID, Source Word, True Translation, Predicted Translation\n")
        for src_wrd in learnt_dict.keys():
            true_and_predicted = learnt_dict[src_wrd]
            f.write(str(bucket_list[src_wrd]) + ", " + str(src_wrd_ids[src_wrd]) + ", " + src_wrd + ", " + str(true_and_predicted['true']).replace(",", "|") + ", " + str(true_and_predicted[
                                                                                                                                                                             'predicted']).replace(",",
                                                                                                                                                                                        "|") + "\n")
