import util
import torch
import numpy as np
import json
import platform
import time
from sklearn.utils.extmath import randomized_svd
import codecs
import scipy

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys


class Evaluator:
    def __init__(self, params, src_emb, tgt_emb, use_cuda=False):
        self.data_dir = params.data_dir
        self.ks = params.ks
        self.methods = params.methods
        self.models = params.models
        self.refine = params.refine
        self.csls_k = params.csls_k
        self.mask_procrustes = params.mask_procrustes
        self.num_refine = params.num_refine

        src = params.src_lang
        tgt = params.tgt_lang

        self.suffix_str = src + '_' + tgt

        self.validation_file = 'validation_' + self.suffix_str + '.npy'
        self.validation_file_new = 'validation_new_' + self.suffix_str + '.npy'
        self.gold_file = 'gold_' + self.suffix_str + '.npy'

        self.tgt_emb = tgt_emb
        self.src_emb = src_emb

        self.valid = []
        self.valid.append(self.prepare_val(self.validation_file))
        self.valid.append(self.prepare_val(self.validation_file_new))

        self.tgt_wrd2id = util.load_npy_one(self.data_dir, "tgt_ids_" + self.suffix_str + ".npy")
        self.tgt_id2wrd = dict(zip(self.tgt_wrd2id.values(), self.tgt_wrd2id.keys()))

        self.r_source = None
        self.r_target = None
        self.r_target = None
        self.use_cuda = use_cuda
        self.cosine_top = params.cosine_top
        self.refine_top = params.refine_top

    def prepare_val(self, validation_file):
        valid = {}
        pass
        valid_dict_ids = util.map_dict2ids(self.data_dir, validation_file, self.suffix_str)
        valid['valid_src_word_ids'] = torch.from_numpy(np.array(list(valid_dict_ids.keys())))
        valid['valid_tgt_word_ids'] = list(valid_dict_ids.values())
        valid['valid_dict'] = util.load_npy_one(self.data_dir, validation_file)
        return valid

    def get_all_precisions(self, mapped_src_emb):
        # Normalize the embeddings
        mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
        tgt_emb = self.tgt_emb / self.tgt_emb.norm(2, 1)[:, None]

        # Calculate r_target
        if 'csls' in self.methods:
            print("Calculating r_target...")
            start_time = time.time()
            self.r_target = common_csls_step(self.csls_k, mapped_src_emb, tgt_emb)
            print("Time taken for making r_target: ", time.time() - start_time)

        adv_mapped_src_emb = mapped_src_emb

        if 'procrustes' in self.models:
            procrustes_mapped_src_emb = self.get_procrustes_mapping()

        if 'with-ref' in self.refine:
            print("Performing refinement...")
            start_time = time.time()
            for _ in range(self.num_refine):
                mapped_src_emb = self.get_refined_mapping(mapped_src_emb, tgt_emb)
                mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
                self.r_target = common_csls_step(self.csls_k, mapped_src_emb, tgt_emb)
            refined_mapped_src_emb = mapped_src_emb
            print("Time taken for refinement: ", time.time() - start_time)

        start_time = time.time()
        all_precisions = {}

        buckets = None
        save = False

        for it, v in enumerate(self.valid):
            if torch.cuda.is_available() and self.use_cuda:
                v['valid_src_word_ids'] = v['valid_src_word_ids'].cuda()
            
            if it == 0:
                key = 'validation'
            else:
                key = 'validation-new'
            all_precisions[key] = {}

            for mod in self.models:
                if mod == 'procrustes':
                    mapped_src_emb = procrustes_mapped_src_emb.clone()
                elif mod == 'adv':
                    mapped_src_emb = adv_mapped_src_emb.clone()
                else:
                    raise 'Model not implemented: %s' % mod
                    
                all_precisions[key][mod] = {}

                for r in self.refine:
                    if r == 'with-ref':
                        if mod == 'procrustes':
                            continue
                        else:
                            mapped_src_emb = refined_mapped_src_emb.clone()
                    
                    mapped_src_emb = mapped_src_emb / mapped_src_emb.norm(2, 1)[:, None]
                    
                    if 'csls' in self.methods:
                        self.r_source = common_csls_step(self.csls_k, tgt_emb, mapped_src_emb[v['valid_src_word_ids']])
                        start_time = time.time()
                        self.r_target = common_csls_step(self.csls_k, mapped_src_emb, tgt_emb)
                        
                    all_precisions[key][mod][r] = {}

                    for m in self.methods:
                        all_precisions[key][mod][r][m] = {}

                        for k in self.ks:
                            if key == 'validation-new' and mod == 'adv' and r == 'with-ref' and m == 'csls' and k == 1:
                                buckets = 5
                                save = True

                            p = self.get_precision_k(k, tgt_emb, mapped_src_emb, v, method=m, buckets=buckets, save=save)
                            if not save:
                                print("key: %s, model: %s, refine: %s, method: %s, k: %d, prec: %f" % (key, mod, r, m, k, p))
                            else:
                                print("key: %s, model: %s, refine: %s, method: %s, k: %d" % (key, mod, r, m, k))
                                print("precision: ", p)
                            all_precisions[key][mod][r][m][k] = p

                            buckets = None
                            save = False
        print("Time taken to run main loop: ", time.time() - start_time)
        print(json.dumps(all_precisions, indent=2))
        return all_precisions

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

    def get_precision_k(self, k, xb, mapped_src_emb, v, method='csls', buckets=None, save=False):
        n = 1500
        xq = mapped_src_emb[v['valid_src_word_ids']]
        tgt_word_ids = v['valid_tgt_word_ids']

        if method == 'nn':
            _, knn_indices = get_knn_indices(k, xb, xq)

        elif method == 'csls':
            knn_indices = csls(k, xb, xq, self.r_source, self.r_target)

        else:
            raise "Method not implemented: %s" % method

        p, c = _calculate_precision(n, knn_indices, tgt_word_ids, buckets)
        if save:
            _save_learnt_dictionary(self.data_dir, v, self.tgt_id2wrd, knn_indices, c)

        return p

    def get_refined_mapping(self, mapped_src_emb, tgt_emb):
        pairs = self.learn_refined_dictionary(mapped_src_emb, tgt_emb)
        return self.do_procrustes(pairs)

    def get_procrustes_mapping(self):
        pairs = self.process_gold_file()
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


def get_knn_indices(k, xb, xq):
    xb = xb.cpu().numpy()
    xq = xq.cpu().numpy()
    d = xq.shape[1]
    if hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        index = faiss.GpuIndexFlatIP(res, d, config)
    else:
        index = faiss.IndexFlatIP(d)

    index.add(xb)
    distances, knn_indices = index.search(xq, k)
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


def common_csls_step(k, xb, xq):
    distances, _ = get_knn_indices(k, xb, xq)
    r = util.to_tensor(np.average(distances, axis=1)).type(torch.FloatTensor)
    return r


def csls(k, xb, xq, r_source, r_target):
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