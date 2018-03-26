import util
from properties import *
from model import *
from evaluation import get_precision_k
from trainer import Trainer
from evaluator import Evaluator
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--data_dir", dest="data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--src_file", dest="src_file", type=str, default=EN_WORD_TO_VEC)
    parser.add_argument("--tgt_file", dest="tgt_file", type=str, default=IT_WORD_TO_VEC)
    parser.add_argument("--validation_file", dest="validation_file", type=str, default=VALIDATION_FILE)

    parser.add_argument("--g_input_size", dest="g_input_size", type=int, default=g_input_size)
    parser.add_argument("--g_output_size", dest="g_output_size", type=int, default=g_output_size)
    parser.add_argument("--d_input_size", dest="d_input_size", type=int, default=d_input_size)
    parser.add_argument("--d_hidden_size", dest="d_hidden_size", type=int, default=d_hidden_size)
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

    parser.add_argument("--iters_in_epoch", dest="iters_in_epoch", type=int, default=iters_in_epoch)
    parser.add_argument("--most_frequent_sampling_size", dest="most_frequent_sampling_size", type=int, default=most_frequent_sampling_size)
    parser.add_argument("--print_every", dest="print_every", type=int, default=print_every)
    parser.add_argument("--lr_decay", dest="lr_decay", type=float, default=lr_decay)
    parser.add_argument("--lr_min", dest="lr_min", type=float, default=lr_min)
    parser.add_argument("--add_noise", dest="add_noise", type=int, default=add_noise)

    parser.add_argument("--noise_mean", dest="noise_mean", type=float, default=noise_mean)
    parser.add_argument("--noise_var", dest="noise_var", type=float, default=noise_var)

    parser.add_argument("--K", dest="K", type=int, default=K)
    parser.add_argument("--top_frequent_words", dest="top_frequent_words", type=int, default=top_frequent_words)
    parser.add_argument("--top_refine", dest="top_refine", type=int, default=top_refine)
    parser.add_argument("--csls_k", dest="csls_k", type=int, default=csls_k)

    parser.add_argument("--mode", dest="mode", type=int, default=mode)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="generator_weights_best_0.t7")

    return parser.parse_args()


def main():
    params = parse_arguments()

    if params.mode == 0:
        u = util.Utils(params)
        u.run()

    else:
        src_emb_array, tgt_emb_array = util.load_npy_two(params.data_dir, 'src.npy', 'tgt.npy')
        src_emb = util.convert_to_embeddings(src_emb_array)
        tgt_emb = util.convert_to_embeddings(tgt_emb_array)
        eval = Evaluator(params, tgt_emb.weight.data, [1, 5, 10], ['nn', 'csls'])

        if params.mode == 1:
            t = Trainer(params)
            g = t.train(src_emb, tgt_emb, eval)

        elif params.mode == 2:
            model_file_path = os.path.join(params.model_dir, params.model_file_name)
            g = Generator(input_size=g_input_size, output_size=g_output_size)
            g.load_state_dict(torch.load(model_file_path, map_location='cpu'))

            if torch.cuda.is_available():
                g = g.cuda()

            eval.get_all_precisions(g(src_emb.weight).data)

        else:
            raise "Invalid flag!"


if __name__ == '__main__':
    main()
