import numpy as np
import torch


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def to_tensor(numpy_array):
    tensor = torch.from_numpy(numpy_array).float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)


# Returns a mapping of words and their embedding
def get_word_vectors(file, top_frequent_words, dir, save=False, save_file_as='en'):
    embeddings = []
    keys = []
    count = 0
    with open(dir + file, 'r', encoding='utf-8') as f:
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
            if count == top_frequent_words:
                break
    np.save(dir + save_file_as + '.npy', np.array(embeddings))
    return np.array(embeddings)


def get_word_vectors_dicts(file, top_frequent_words, dir, save=False, save_file_as='en_dict'):
    word2vec = {}
    count = 0
    with open(dir + file, 'r', encoding='utf-8') as f:
        ignore_first_row = True
        for row in f.readlines():
            if ignore_first_row:
                ignore_first_row = False
                continue
            split_row = row.split(" ")
            vec = np.array(split_row[1:-1]).astype(np.float)
            if len(vec) == 300:
                word2vec[split_row[0]] = vec
            count += 1
            if count == top_frequent_words:
                break
    if save:
        np.save(dir + save_file_as + '.npy', word2vec)
    return word2vec


def get_validation_set(file, dir, save=False, save_file_as='validation'):
    true_dict = {}
    with open(dir + file, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            split_row = row.split(" ")
            key = split_row[0]
            value = split_row[1].rstrip("\n")
            if key not in true_dict.keys():
                true_dict[key] = []
            true_dict[split_row[0]].append(value)
    if save:
        np.save(dir + save_file_as + '.npy', true_dict)
    return true_dict


def convert_to_embeddings(emb_array):
    emb_tensor = to_tensor(emb_array)
    v, d = emb_tensor.size()
    emb = torch.nn.Embedding(v, d)
    if torch.cuda.is_available():
        emb = emb.cuda()
    emb.weight.data.copy_(emb_tensor)
    emb.weight.requires_grad = False
    return emb

# Before using this method make sure you run this util file once to create the data files en.npy and it.npy
# Returns the monolingual embeddings in en and it


def get_embeddings(dir):
    return np.load(dir + 'en.npy'), np.load(dir + 'it.npy')


def get_embeddings_dicts(dir):
    return np.load(dir + 'en_dict.npy').item(), np.load(dir + 'it_dict.npy').item()


def get_true_dict(dir):
    return np.load(dir + 'validation.npy').item()


def run(params):
    data_dir = params.data_dir
    src_file = params.src_file
    tgt_file = params.tgt_file
    validation_file = params.validation_file
    top_frequent_words = params.top_frequent_words

    print("Reading english word embeddings...")
    word2vec_en = get_word_vectors(src_file, top_frequent_words, data_dir, save=True, save_file_as='en')
    print(word2vec_en.shape)

    print("Reading italian word embeddings...")
    word2vec_it = get_word_vectors(tgt_file, top_frequent_words, data_dir, save=True, save_file_as='it')
    print(word2vec_it.shape)

    print("Creating word vectors for both languages...")
    word2vec_en = get_word_vectors_dicts(src_file, top_frequent_words, data_dir, save=True, save_file_as='en_dict')
    word2vec_it = get_word_vectors_dicts(tgt_file, top_frequent_words, data_dir, save=True, save_file_as='it_dict')

    print("Reading validation file...")
    true_dict = get_validation_set(validation_file, data_dir, save=True)

    print("Done !!")
