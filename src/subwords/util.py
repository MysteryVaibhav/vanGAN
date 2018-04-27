from collections import defaultdict
import numpy as np
import torch
import random

from embeddings import Embedding
from embeddings import SubwordEmbedding

seed = 0
np.random.seed(seed)
random.seed(seed)


class Utils:

    def __init__(self, params):
        self.params = params
        self.data_dir = params.data_dir
        self.src_file = params.src_file
        self.tgt_file = params.tgt_file
        self.validation_file = params.validation_file
        self.full_file = params.full_file
        self.new_validation_file = params.new_validation_file
        self.gold_file = params.gold_file
        self.top_frequent_words = params.top_frequent_words

    def run(self):
        src = self.params.src_lang
        tgt = self.params.tgt_lang

        suffix_str = src + '_' + tgt
        print("Reading source word embeddings...")
        word2vec_src = self.save_word_vectors(self.src_file, save=True, save_file_as='src_' + suffix_str)
        print("Done.")
        print(word2vec_src.shape)
        print("Reading target word embeddings...")
        word2vec_tgt = self.save_word_vectors(self.tgt_file, save=True, save_file_as='tgt_' + suffix_str)
        print("Done.")
        print(word2vec_tgt.shape)
        print("Reading validation file...")
        self.read_dictionary(self.validation_file, save_file_as="validation_" + suffix_str, save=True)
        print("Reading gold file...")
        self.read_dictionary(self.gold_file, save_file_as='gold_' + suffix_str, save=True)
        print("Constructing source word-id map...")
        self.save_word_ids_dicts(self.src_file, save=True, save_file_as='src_ids_' + suffix_str)
        print("Done.")
        print("Constructing target word-id map...")
        self.save_word_ids_dicts(self.tgt_file, save=True, save_file_as='tgt_ids_' + suffix_str)

        # print("Reading full file...")
        # full_dict = self.read_dictionary(self.full_file, save=False)
        # all_src_words = list(full_dict.keys())
        # word2id = dict(zip(np.arange(len(all_src_words)), all_src_words))
        # print("Constructing new validation set...")
        # self.construct_new_val_set(full_dict, word2id, self.new_validation_file)
        self.read_dictionary(self.new_validation_file, save_file_as="validation_new_" + suffix_str, save=True)
        print("Everything Done.")

    def save_word_vectors(self, file, save=False, save_file_as='src'):
        embeddings = []
        keys = []
        count = 0
        with open(self.data_dir + file, 'r', encoding='utf-8') as f:
            ignore_first_row = True
            for row in f.readlines():
                if ignore_first_row:
                    ignore_first_row = False
                    continue
                split_row = row.split(" ")
                vec = np.array(split_row[1:-1]).astype(np.float)
                if len(vec) == 300:
                    embeddings.append(vec)
                    keys.append(split_row[0])
                count += 1
                if count == self.top_frequent_words:
                    break
        if save:
            np.save(self.data_dir + save_file_as + '.npy', np.array(embeddings))
        return np.array(embeddings)

    def save_word_ids_dicts(self, file, save=False, save_file_as='src_ids'):
        word2id = {}
        count = 0
        with open(self.data_dir + file, 'r', encoding='utf-8') as f:
            ignore_first_row = True
            for row in f.readlines():
                if ignore_first_row:
                    ignore_first_row = False
                    continue
                split_row = row.split(" ")
                vec = np.array(split_row[1:-1]).astype(np.float)
                if len(vec) == 300:
                    word2id[split_row[0]] = count
                count += 1
                if count == self.top_frequent_words:
                    break
        if save:
            np.save(self.data_dir + save_file_as + '.npy', word2id)
        return word2id

    def read_dictionary(self, file, save=False, save_file_as='validation'):
        true_dict = {}
        with open(self.data_dir + file, 'r', encoding='utf-8') as f:
            rows = f.readlines()
            for row in rows:
                split_row = row.split(" ")
                key = split_row[0]
                value = split_row[1].rstrip("\n")
                if key not in true_dict.keys():
                    true_dict[key] = []
                true_dict[split_row[0]].append(value)
        if save:
            np.save(self.data_dir + save_file_as + '.npy', true_dict)
        return true_dict

    def construct_new_val_set(self, full_dict, word2id, fname):
        n = len(list(word2id.keys()))
        indices = []
        buckets = 5
        num_per_bucket = int(1500/buckets)

        for i in range(buckets):
            lo = int(i * n/buckets)
            hi = int((i+1) * n/buckets)
            indices.extend(random.sample(range(lo, hi, 1), num_per_bucket))

        indices = sorted(indices)
        all_words = []
        with open(self.data_dir + fname, 'w', encoding='utf-8') as f:
            for i in indices:
                wrd = word2id[i]
                all_words.append(wrd)
                for tgt in full_dict[wrd]:
                    f.write(wrd + " " + tgt + "\n")


def load_npy_one(data_dir, fname):
    return np.load(data_dir + fname).item()

def pad(seqs, length=None, value=0):
    """Pad a sequence with `value`.

    Args:
    - seqs: a list of sequences
    - length: if None, use max length
    - value: filler
    """
    if length is None:
        length = max([len(seq) for seq in seqs])
    seqs_ = [seq + [value for _ in range(length - len(seq))]
             for seq in seqs]
    return [seq[:length] for seq in seqs_]
    

def load_subword_embeddings(filename):
    """Load subword embeddings from a .npz file.
    """
    if not filename.endswith('.npz'):
        msg = 'Pretrained word embeddings must be in .npz'
        mst += ' Received ' + filename
        raise ValueError(msg)
    data = dict(np.load(filename))
    N = data['seqs'].shape[0]
    D = data['W'].shape[1]
    W = np.r_[np.zeros((1, D)), data['W']]
    seqs = [[v + 1 for v in seq] for seq in data['seqs']]  # 0=PAD
    idx2id = [0] + list(data['idx2id'])  # 0=PAD
    return {'E': Embedding(W),
            'F': SubwordEmbedding(D, n_layers=1),
            'vecs': torch.FloatTensor(np.empty((N, D))),
            'seqs': torch.LongTensor(pad(seqs)),
            'idx2id': idx2id,
            'id2idx': {i: idx for idx, i in enumerate(idx2id)}}


def load_word_embeddings(filename):
    """Load word embeddings from a .txt file.
    """
    if not filename.endswith('.npz'):
        msg = 'Pretrained word embeddings must be in .npz'
        mst += ' Received ' + filename
        raise ValueError(msg)
    data = dict(np.load(filename))
    return {'E': Embedding(data['W']),
            'idx2word': data['idx2word'],
            'word2idx': {i: word for word, i in enumerate(data['idx2word'])}}


def read_validation_file(filename, src_indexer, tgt_indexer):
    """Read a validation file.

    Args:
    - filename: path to a validation file
      `File format: <src:subword IDs>\t<tgt:subword IDs>`
    - indexer: map subword ID to an index starting from zero
    """
    src_idx, src_seqs, tgt_idx = [], [], []
    w2i = defaultdict(lambda: len(w2i))
    with open(filename) as f:
        for line in f:
            row = line.rstrip('\n').split('\t')
            src = [int(subword) for subword in row[2].split()]
            tgt = row[1]
            try:
                src_seqs_ = [src_indexer[subword] for subword in src]
                tgt_idx_ = tgt_indexer[tgt]
            except KeyError:
                continue
            src_idx.append(w2i[row[0]])  # record source word IDs (for aggregation)
            src_seqs.append(src_seqs_)
            tgt_idx.append(tgt_idx_)
    return np.array(src_idx), np.array(src_seqs), np.array(tgt_idx)


def drop_oov_from_validation_set(src_seqs, tgt_indices, src_n_vocab, tgt_n_vocab):
    """Drop a translation pair that contains OOV.

    Args:
    - src, tgt: a list of (sub)word IDs
    - src_vocab, tgt_vocab: vocabulary of subwords
    """
    indices = []
    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_indices)):
        # If a sequence contains OOV words, skip
        if any([idx >= src_n_vocab for idx in src]):
            continue
        if tgt >= tgt_n_vocab:
            continue
        indices.append(i)  # memo an index
    print('Validation: {} instances'.format(len(indices)))

    return src_seqs[indices], tgt_indices[indices]


 
# Validation set in a dictionary form {src_wrd: [tgt_wrd_1, tgt_wrd_2, ...]}
def map_dict2ids(data_dir, dict_fname, suffix_str):
    dict_wrd = load_npy_one(data_dir, dict_fname)
    src_ids, tgt_ids = load_npy_two(data_dir, 'src_ids_' + suffix_str + '.npy', 'tgt_ids_' + suffix_str +'.npy', dict=True)
    dict_ids = {}
    for src_wrd, tgt_list in dict_wrd.items():
        dict_ids[src_ids[src_wrd]] = [tgt_ids[tgt_wrd] for tgt_wrd in tgt_list]
    return dict_ids


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def to_tensor(numpy_array):
    """Convert numpy.ndarray into FloatTensor."""
    tensor = torch.from_numpy(numpy_array).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_variable(tensor, volatile=False, use_cuda=True):
    if torch.cuda.is_available() and use_cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)

