import util
from properties import *
from model import *
from trainer import Trainer, get_hyperparams, construct_input
from evaluator import Evaluator
import argparse
import copy
import os
import numpy as np
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--data_dir", dest="data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--src_file", dest="src_file", type=str, default=EN_WORD_TO_VEC)
    parser.add_argument("--tgt_file", dest="tgt_file", type=str, default=IT_WORD_TO_VEC)
    parser.add_argument("--validation_file", dest="validation_file", type=str, default=VALIDATION_FILE)
    parser.add_argument("--full_file", dest="full_file", type=str, default=FULL_FILE)
    parser.add_argument("--new_validation_file", dest="new_validation_file", type=str, default=NEW_VAL_FILE)
    parser.add_argument("--gold_file", dest="gold_file", type=str, default=GOLD_FILE)

    parser.add_argument("--g_input_size", dest="g_input_size", type=int, default=g_input_size)
    parser.add_argument("--g_output_size", dest="g_output_size", type=int, default=g_output_size)
    parser.add_argument("--d_input_size", dest="d_input_size", type=int, default=d_input_size)
    parser.add_argument("--d_hidden_size", dest="d_hidden_size", type=int, default=d_hidden_size)
    parser.add_argument("--g_hidden_size", dest="g_hidden_size", type=int, default=g_hidden_size)
    parser.add_argument("--d_output_size", dest="d_output_size", type=int, default=d_output_size)
    parser.add_argument("--mini_batch_size", dest="mini_batch_size", type=int, default=mini_batch_size)

    parser.add_argument("--d_learning_rate", dest="d_learning_rate", type=float, default=d_learning_rate)
    parser.add_argument("--g_learning_rate", dest="g_learning_rate", type=float, default=g_learning_rate)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=num_epochs)
    parser.add_argument("--d_steps", dest="d_steps", type=int, default=d_steps)
    parser.add_argument("--g_steps", dest="g_steps", type=int, default=g_steps)
    parser.add_argument("--smoothing", dest="smoothing", type=float, default=smoothing)
    parser.add_argument("--beta", dest="beta", type=float, default=beta)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=clip_value)
    parser.add_argument("--num_random_seeds", dest="num_random_seeds", type=int, default=num_random_seeds)
    parser.add_argument("--k_neighbours_inp", dest="k_neighbours_inp", type=int, default=k_neighbours_inp)

    parser.add_argument("--iters_in_epoch", dest="iters_in_epoch", type=int, default=iters_in_epoch)
    parser.add_argument("--most_frequent_sampling_size", dest="most_frequent_sampling_size", type=int, default=most_frequent_sampling_size)
    parser.add_argument("--print_every", dest="print_every", type=int, default=print_every)
    parser.add_argument("--lr_decay", dest="lr_decay", type=float, default=lr_decay)
    parser.add_argument("--lr_min", dest="lr_min", type=float, default=lr_min)
    parser.add_argument("--center_embeddings", dest="center_embeddings", type=int, default=center_embeddings)

    parser.add_argument("--dropout_inp", dest="dropout_inp", type=int, default=dropout_inp)
    parser.add_argument("--dropout_hidden", dest="dropout_hidden", type=int, default=dropout_hidden)
    parser.add_argument("--leaky_slope", dest="leaky_slope", type=int, default=leaky_slope)

    parser.add_argument("--add_noise", dest="add_noise", type=int, default=add_noise)
    parser.add_argument("--noise_mean", dest="noise_mean", type=float, default=noise_mean)
    parser.add_argument("--noise_var", dest="noise_var", type=float, default=noise_var)

    parser.add_argument("--K", dest="K", type=int, default=K)
    parser.add_argument("--top_frequent_words", dest="top_frequent_words", type=int, default=top_frequent_words)

    parser.add_argument("--csls_k", dest="csls_k", type=int, default=csls_k)

    parser.add_argument("--mode", dest="mode", type=int, default=mode)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="generator_weights_best_0.t7")

    parser.add_argument("--refine_top", dest="refine_top", type=int, default=refine_top)
    parser.add_argument("--cosine_top", dest="cosine_top", type=int, default=cosine_top)
    parser.add_argument("--mask_procrustes", dest="mask_procrustes", type=int, default=0)
    parser.add_argument("--num_refine", dest="num_refine", type=int, default=1)
    parser.add_argument("--context", dest="context", type=int, default=context)
    parser.add_argument("--atype", dest="atype", type=str, default=atype)

    parser.add_argument("--src_lang", dest="src_lang", type=str, default='en')
    parser.add_argument("--tgt_lang", dest="tgt_lang", type=str, default='zh')
    return parser.parse_args()


def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1, 5, 10]
    params.methods = ['nn', 'csls']
    params.models = ['procrustes', 'adv']
    params.refine = ['without-ref', 'with-ref']
    return params


def main():
    params = parse_arguments()

    if params.mode == 0:
        u = util.Utils(params)
        u.run()

    else:
        print("Reading embedding numpy files...")
        use_cuda = False
        if params.mode == 1:
            use_cuda = True

        src = params.src_lang
        tgt = params.tgt_lang

        suffix_str = src + '_' + tgt

        src_emb_array, tgt_emb_array = util.load_npy_two(params.data_dir, 'src_' + suffix_str + '.npy', 'tgt_' + suffix_str + '.npy')
        print("Done.")
        print("Converting arrays to embedding layers...")
        src_emb = util.convert_to_embeddings(src_emb_array, use_cuda)
        tgt_emb = util.convert_to_embeddings(tgt_emb_array, use_cuda)
        print("Done.")
        
        if params.center_embeddings > 0:
            util.center_embeddings(src_emb.weight.data)
            util.center_embeddings(tgt_emb.weight.data)
        
        if params.mode == 1:
            t = Trainer(params)
            g = t.train(src_emb, tgt_emb)

        elif params.mode == 2:
            params = _get_eval_params(params)
            evaluator = Evaluator(params, src_emb.weight.data, tgt_emb.weight.data)

            model_file_path = os.path.join(params.model_dir, params.model_file_name)
            g = Generator(input_size=params.g_input_size, hidden_size=params.g_hidden_size,
                          output_size=params.g_output_size, hyperparams=get_hyperparams(params, disc=False))
            g.load_state_dict(torch.load(model_file_path, map_location='cpu'))

            try:
                knn_list = pickle.load(open('full_knn_list_' + suffix_str + '.pkl', 'rb'))
            except FileNotFoundError:
                print("k-nn file not found!")
            knn_emb = util.convert_to_embeddings(knn_list, use_cuda=False)

            attn = Attention(atype=params.atype)
            indices = torch.arange(params.top_frequent_words).type(torch.LongTensor)

            if params.context == 1:
                mapped_src_emb = g(construct_input(knn_emb, indices, src_emb, attn)).data
            else:
                mapped_src_emb = g(src_emb.weight).data

#             if torch.cuda.is_available():
#                 g = g.cuda()


#             print(mapped_src_emb)
            evaluator.get_all_precisions(mapped_src_emb)
            # print("Unsupervised criterion: ", evaluator.calc_unsupervised_criterion(mapped_src_emb))

            # unsupervised_criterion = []
            #
            # for i in range(40):
            #     model_file_path = os.path.join(params.model_dir, 'generator_weights_en_es_' + str(i+1) + '.t7')
            #     g = Generator(input_size=g_input_size, output_size=g_output_size)
            #     g.load_state_dict(torch.load(model_file_path, map_location='cpu'))
            #     if torch.cuda.is_available():
            #         g = g.cuda()
            #     mapped_src_emb = g(src_emb.weight).data
            #     uc = evaluator.calc_unsupervised_criterion(mapped_src_emb)
            #     print("i: %d, uc: %f" % (i, uc))
            #     unsupervised_criterion.append(uc)
            #
            # np.save("uc.npy", np.array(unsupervised_criterion))

        else:
            raise "Invalid flag!"


if __name__ == '__main__':
    main()
