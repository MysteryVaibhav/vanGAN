from os import path
import argparse
import json
import numpy as np

from evaluator import Evaluator
from model import *
from properties import *
from trainer import Trainer
import util


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for GAN by Zhang et al. (ACL2017)')
    parser.add_argument('--data_dir', dest='data_dir',
                        type=str, default=DATA_DIR)
    parser.add_argument('--src_file', dest='src_file',
                        type=str, default=EN_WORD_TO_VEC)
    parser.add_argument('--tgt_file', dest='tgt_file',
                        type=str, default=IT_WORD_TO_VEC)
    parser.add_argument('--src_freq_file', dest='src_freq_file',
                        type=str, default=EN_WORD_FREQ)
    parser.add_argument('--tgt_freq_file', dest='tgt_freq_file',
                        type=str, default=IT_WORD_FREQ)
    parser.add_argument('--validation_file', dest='validation_file',
                        type=str, default=VALIDATION_FILE)
    parser.add_argument('--g_input_size', dest='g_input_size',
                        type=int, default=g_input_size)
    parser.add_argument('--g_output_size', dest='g_output_size',
                        type=int, default=g_output_size)
    parser.add_argument('--d_input_size', dest='d_input_size',
                        type=int, default=d_input_size)
    parser.add_argument('--d_hidden_size', dest='d_hidden_size',
                        type=int, default=d_hidden_size)
    parser.add_argument('--d_output_size', dest='d_output_size',
                        type=int, default=d_output_size)
    parser.add_argument('--mini_batch_size', dest='mini_batch_size',
                        type=int, default=mini_batch_size)
    parser.add_argument('--d_learning_rate', dest='d_learning_rate',
                        type=float, default=d_learning_rate)
    parser.add_argument('--g_learning_rate', dest='g_learning_rate',
                        type=float, default=g_learning_rate)
    parser.add_argument('--uniform-sampling', action='store_true',
                        help='uniformly sample embeddings in training')
    parser.add_argument('--max-iters', type=int, default=max_iters)
    parser.add_argument('--most_frequent_sampling_size', dest='most_frequent_sampling_size', type=int, default=most_frequent_sampling_size)
    parser.add_argument('--top_frequent_words', dest='top_frequent_words', type=int, default=top_frequent_words)
    parser.add_argument('--gan-model', type=int, default=3,
                        help='GAN Model {1,2,3} (default=3)')
    parser.add_argument('--lambda-r', type=float, default=1.0,
                        help='Coefficient for a reconstruction loss')
    parser.add_argument('--csls_k', dest='csls_k', type=int, default=csls_k)
    parser.add_argument('--mode', dest='mode', type=int, default=1)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--model_file_name', dest='model_file_name', type=str)
    args = parser.parse_args()
    if args.model_file_name is None:
        args.model_file_name = 'g{}_best.pth'.format(args.gan_model)
    return args


def main():
    params = parse_arguments()

    if params.mode == 0:  # Preparing word vectors
        u = util.Utils(params)
        u.run()
        return

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    src_emb_array, tgt_emb_array = util.load_npy_two(params.data_dir, 'src.npy', 'tgt.npy')
    src_emb = util.convert_to_embeddings(src_emb_array)
    tgt_emb = util.convert_to_embeddings(tgt_emb_array)
    src_emb.weight.data /= src_emb.weight.data.norm(2, dim=1).view((-1, 1))
    tgt_emb.weight.data /= tgt_emb.weight.data.norm(2, dim=1).view((-1, 1))
    evaluator = Evaluator(params, tgt_emb.weight.data, [1, 5, 10], ['nn', 'csls'])

    if params.mode == 1:
        t = Trainer(params)
        g = t.train(src_emb, tgt_emb, evaluator)

    elif params.mode == 2:
        model_file_path = path.join(params.model_dir, params.model_file_name)
        g = Generator(input_size=g_input_size, output_size=g_output_size)
        g.load_state_dict(torch.load(model_file_path, map_location='cpu'))

        if torch.cuda.is_available():
            g = g.cuda()

        all_precisions = evaluator.get_all_precisions(g(src_emb.weight).data)
        print(json.dumps(all_precisions))

    else:
        raise "Invalid flag!"


if __name__ == '__main__':
    main()
