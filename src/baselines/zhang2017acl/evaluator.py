import faiss
import json
import numpy as np
import time
import torch

import util


class Evaluator:
    def __init__(self, params, tgt_emb, k, methods):
        self.data_dir = params.data_dir
        self.validation_file = 'validation.npy'
        self.k = k
        self.methods = methods
        self.csls_k = params.csls_k

        self.valid_dict = util.load_npy_one(self.data_dir, self.validation_file)
        self.valid_dict_ids = util.get_validation_set_ids(self.data_dir, self.validation_file)
        self.valid_src_word_ids = torch.from_numpy(np.array(list(self.valid_dict_ids.keys())))
        if torch.cuda.is_available():
            self.valid_src_word_ids = self.valid_src_word_ids.cuda()
        self.valid_tgt_word_ids = list(self.valid_dict_ids.values())
        self.tgt_emb = tgt_emb / tgt_emb.norm(2, 1)[:, None]
        self.num_valid = self.valid_src_word_ids.cpu().numpy().shape[0]

        self.r_source = None
        self.r_target = None

    def get_all_precisions(self, mapped_src_emb):
        mapped_src_emb = mapped_src_emb/mapped_src_emb.norm(2, 1)[:, None]
        all_precisions = {}

        start = time.time()
        if 'csls' in self.methods:
            self.r_source, self.r_target = self.common_csls_step(mapped_src_emb)

        for m in self.methods:
            all_precisions[m] = {}
            for k in self.k:
                all_precisions[m][k] = self.get_precision_k(k, mapped_src_emb, method=m)
        return all_precisions

    def common_csls_step(self, mapped_src_emb):
        xq = mapped_src_emb[self.valid_src_word_ids]
        xb = self.tgt_emb
        distances, _ = _get_knn_indices(self.csls_k, xb, xq)
        r_source = util.to_tensor(np.average(distances, axis=1))
        distances, _ = _get_knn_indices(self.csls_k, mapped_src_emb, xb)
        r_target = util.to_tensor(np.average(distances, axis=1))
        return r_source, r_target

    def get_precision_k(self, k, mapped_src_emb, method='csls'):
        xq = mapped_src_emb[self.valid_src_word_ids]
        xb = self.tgt_emb

        if method == 'nn':
            _, knn_indices = _get_knn_indices(k, xb, xq)

        elif method == 'csls':
            knn_indices = self.csls(k, xb, xq)

        else:
            raise "Method not implemented: %s" % method

        p = round(self.calculate_precision(knn_indices), 2)
        return p

    def csls(self, k, xb, xq):
        csls = 2 * xq.mm(xb.transpose(0, 1))
        csls.sub_(self.r_source[:, None] + self.r_target[None, :])
        knn_indices = csls.topk(k, dim=1)[1]
        return knn_indices

    def calculate_precision(self, knn_indices):
        p = 0.0
        for i, knn in enumerate(knn_indices):
            if len(set(knn).intersection(set(self.valid_tgt_word_ids[i]))) > 0:
                p += 1.0
        return (p/self.num_valid)*100


def _get_knn_indices(k, xb, xq):
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
