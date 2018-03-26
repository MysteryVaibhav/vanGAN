import util
import torch
import numpy as np
import json
import platform
import time

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys


class Evaluator:
    def __init__(self, params, tgt_emb, k, methods):
        self.data_dir = params.data_dir
        self.validation_file = 'validation.npy'
        self.validation_file_new = 'validation_new.npy'
        self.k = k
        self.methods = methods
        self.csls_k = params.csls_k

        self.tgt_emb = tgt_emb / tgt_emb.norm(2, 1)[:, None]

        self.valid = []
        self.valid.append(self.prepare_val(self.validation_file))
        self.valid.append(self.prepare_val(self.validation_file_new))

        self.r_source = None
        self.r_target = None
        self.top_freq = 10000

    def prepare_val(self, validation_file):
        valid = {}
        pass
        valid_dict_ids = util.get_validation_set_ids(self.data_dir, validation_file)
        valid['valid_src_word_ids'] = torch.from_numpy(np.array(list(valid_dict_ids.keys())))
        valid['valid_tgt_word_ids'] = list(valid_dict_ids.values())
        valid['valid_dict'] = util.load_npy_one(self.data_dir, validation_file)
        return valid

    def get_all_precisions(self, mapped_src_emb):
        mapped_src_emb = mapped_src_emb/mapped_src_emb.norm(2, 1)[:, None]
        all_precisions = {}

        start = time.time()

        for it, v in enumerate(self.valid):
            all_precisions[it] = {}

            if 'csls' in self.methods:
                self.r_source, self.r_target = self.common_csls_step(mapped_src_emb, v['valid_src_word_ids'])

            for m in self.methods:
                all_precisions[it][m] = {}
                for k in self.k:
                    all_precisions[it][m][k] = self.get_precision_k(k, mapped_src_emb, v['valid_src_word_ids'], v['valid_tgt_word_ids'], method=m)

        print(json.dumps(all_precisions, indent=2))
        return all_precisions

    def calc_unsupervised_criterion(self, mapped_src_emb):
        src_wrd_ids = torch.arange(self.top_freq).type(torch.LongTensor)
        start_time = time.time()
        r_source, r_target = self.common_csls_step(mapped_src_emb, src_wrd_ids)
        print("Time taken: ", time.time()-start_time)
        xq = mapped_src_emb[src_wrd_ids]
        xb = self.tgt_emb
        knn_indices = self.csls(1, xb, xq, r_source, r_target)
        knn_indices = knn_indices.view(knn_indices.numel())
        sim = xq.mm(self.tgt_emb[knn_indices].transpose(0, 1))
        print(sim.mean())
        return sim.mean()

    def common_csls_step(self, mapped_src_emb, src_wrd_ids):
        xq = mapped_src_emb[src_wrd_ids]
        xb = self.tgt_emb
        distances, _ = _get_knn_indices(self.csls_k, xb, xq)
        r_source = util.to_tensor(np.average(distances, axis=1))
        distances, _ = _get_knn_indices(self.csls_k, mapped_src_emb, xb)
        r_target = util.to_tensor(np.average(distances, axis=1))
        return r_source, r_target

    def get_precision_k(self, k, mapped_src_emb, src_wrd_ids, tgt_word_ids, method='csls'):
        xq = mapped_src_emb[src_wrd_ids]
        n = src_wrd_ids.size()[0]
        xb = self.tgt_emb

        if method == 'nn':
            _, knn_indices = _get_knn_indices(k, xb, xq)

        elif method == 'csls':
            knn_indices = self.csls(k, xb, xq)

        else:
            raise "Method not implemented: %s" % method

        p = round(self.calculate_precision(knn_indices, n, tgt_word_ids), 2)
        print(p)
        return p

    def csls(self, k, xb, xq, r_source=None, r_target=None):
        if r_source is None and r_target is None:
            r_source = self.r_source
            r_target = self.r_target
        csls = 2 * xq.mm(xb.transpose(0, 1))
        csls.sub_(r_source[:, None] + r_target[None, :])
        knn_indices = csls.topk(k, dim=1)[1]
        return knn_indices

    def calculate_precision(self, knn_indices, n, tgt_word_ids):
        p = 0.0
        for i, knn in enumerate(knn_indices):
            if len(set(knn).intersection(set(tgt_word_ids[i]))) > 0:
                p += 1.0
        return (p/n)*100


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
