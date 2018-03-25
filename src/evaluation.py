import platform
from util import *
from properties import *
import os

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_precision(true_dict, predicted_dict):
    """Calculates precision given true and predicted dictionaries
    Input:
        true_dict - true dictionary of words to possible translations
        predicted_dict - predicted dictionary of words to possible translations
    Output:
        Precision value
    """
    total_correct = 0
    for (word, translation) in predicted_dict.items():
        true_translations = set(true_dict[word])
        predicted_translations = set(translation)
        if len(true_translations.intersection(predicted_translations)) > 0:
            total_correct += 1
    return float(total_correct)/len(predicted_dict.keys())


def get_translation_dict(source_word_list, target_word_list, knn_indices):
    translation_dict = {}
    for (i, word) in enumerate(source_word_list):
        # print("%d: %s" % (i, word))
        translation_dict[word] = [target_word_list[j] for j in list(knn_indices[i])]
    #print(json.dumps(translation_dict, indent=2))
    return translation_dict


def get_knn_indices(k, xb, xq):
    if hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        index = faiss.GpuIndexFlatIP(res, g_input_size, config)
    else:
        index = faiss.IndexFlatIP(g_input_size)

    index.add(xb)
    distances, knn_indices = index.search(xq, k)
    return distances, knn_indices


def CSLS_fast(k, xb, xq):
    distances, _ = get_knn_indices(csls_k, xb, xq)
    r_source = np.average(distances, axis=1)
    distances, _ = get_knn_indices(csls_k, xq, xb)
    r_target = np.average(distances, axis=1)

    n_source = np.shape(r_source)[0]
    n_target = np.shape(r_target)[0]

    knn_indices = []
    for i in range(n_source):
        src_wemb = xq[i, :]
        c = np.sum(np.multiply(np.repeat(src_wemb[np.newaxis, :],  n_target, axis=0), xb), axis=1)
        rs = np.repeat(r_source[i],  n_target, axis=0)
        csls = 2*c - rs - r_target
        knn_indices.append(np.argsort(csls)[-k:])

    return knn_indices


def CSLS_faster(k, xb, xq, tgt_emb, src_emb):
    distances, _ = get_knn_indices(csls_k, xb, xq)
    r_source = np.average(distances, axis=1)
    distances, _ = get_knn_indices(csls_k, xq, xb)
    r_target = np.average(distances, axis=1)

    n_source = np.shape(r_source)[0]
    n_target = np.shape(r_target)[0]

    knn_indices = []
    for i in range(n_source):
        src_wemb = xq[i, :]
        c = np.sum(np.multiply(np.repeat(src_wemb[np.newaxis, :],  n_target, axis=0), xb), axis=1)
        rs = np.repeat(r_source[i],  n_target, axis=0)
        csls = 2*c - rs - r_target
        knn_indices.append(np.argsort(csls)[-k:])

    return knn_indices


def cosine_similarity(m1, m2):
    numerator = np.sum(np.multiply(m1, m2), axis=1)
    denominator = np.sqrt(np.multiply(np.sum(np.multiply(m1, m1), axis=1),
                                      np.sum(np.multiply(m2, m2), axis=1)))
    return numerator/denominator


def get_mapped_embeddings(g, data_dir, source_word_list):
    source_vec_dict, target_vec_dict = get_embeddings_dicts(data_dir)
    target_word_list = list(target_vec_dict.keys())

    # word_tensors = to_tensor(np.array([source_vec_dict[source_word] for source_word in source_word_list]).astype(float))
    # mapped_embeddings = g(to_variable(word_tensors)).data.cpu().numpy()

    mapped_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        word_tensor = to_tensor(np.array(source_vec_dict[source_word]).astype(float))
        mapped_embeddings[i] = g(to_variable(word_tensor)).data.cpu().numpy()
    return mapped_embeddings, target_word_list


def get_precision_k(k, g, true_dict, data_dir, method='csls'):
    source_word_list = true_dict.keys()

    _, xb = get_embeddings(data_dir)
    xb = np.float32(xb)
    row_sum = np.linalg.norm(xb, axis=1)
    xb = xb/row_sum[:, np.newaxis]

    xq, target_word_list = get_mapped_embeddings(g, data_dir, source_word_list)
    xq = np.float32(xq)
    row_sum = np.linalg.norm(xq, axis=1)
    xq = xq/row_sum[:, np.newaxis]

    if method == 'nn':
        _, knn_indices = get_knn_indices(k, xb, xq)
    elif method == 'csls':
        knn_indices = CSLS_fast(k, xb, xq)
    elif method == 'csls_faster':
        src_emb = convert_to_embeddings(xq)
        tgt_emb = convert_to_embeddings(xb)
        knn_indices = CSLS_faster(k, xb, xq, tgt_emb, src_emb)
    else:
        raise 'Method not implemented: %s' % method

    predicted_dict = get_translation_dict(source_word_list, target_word_list,
                                          knn_indices)
    return calculate_precision(true_dict, predicted_dict)


def test_function(source_word_list):
    source_vec_dict, _ = get_embeddings_dicts()
    source_embeddings = np.zeros((len(source_word_list), g_input_size))
    for (i, source_word) in enumerate(source_word_list):
        source_embeddings[i] = source_vec_dict[source_word]
    return source_embeddings