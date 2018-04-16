import numpy as np
import torch
import random

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


def load_npy_two(data_dir, src_fname, tgt_fname, dict=False):
    if dict:
        x = np.load(data_dir + src_fname).item()
        y = np.load(data_dir + tgt_fname).item()
    else:
        x = np.load(data_dir + src_fname)
        y = np.load(data_dir + tgt_fname)
    return x, y


# Validation set in a dictionary form {src_wrd: [tgt_wrd_1, tgt_wrd_2, ...]}
def map_dict2ids(data_dir, dict_fname, suffix_str):
    dict_wrd = load_npy_one(data_dir, dict_fname)
    src_ids, tgt_ids = load_npy_two(data_dir, 'src_ids_' + suffix_str + '.npy', 'tgt_ids_' + suffix_str +'.npy', dict=True)
    dict_ids = {}
    for src_wrd, tgt_list in dict_wrd.items():
        dict_ids[src_ids[src_wrd]] = [tgt_ids[tgt_wrd] for tgt_wrd in tgt_list]
    return dict_ids


def convert_to_embeddings(emb_array, use_cuda=False):
    emb_tensor = to_tensor(emb_array)
    v, d = emb_tensor.size()
    emb = torch.nn.Embedding(v, d)
    if torch.cuda.is_available() and use_cuda:
        emb = emb.cuda()
    emb.weight.data.copy_(emb_tensor)
    emb.weight.requires_grad = False
    return emb


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def center_embeddings(emb):
    mean = emb.mean(0, keepdim=True).expand_as(emb)
    emb.sub_(mean)


def to_tensor(numpy_array):
    tensor = torch.from_numpy(numpy_array).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_variable(tensor, volatile=False, use_cuda=True):
    if torch.cuda.is_available() and use_cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)
