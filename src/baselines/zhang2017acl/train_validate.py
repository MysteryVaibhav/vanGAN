import argparse
import os
import platform
import torch

from properties import *
from trainer import train
from util import *
from evaluate import get_precision_k

op_sys = platform.system()
if op_sys == 'Darwin':
    from faiss_master import faiss
elif op_sys == 'Linux':
    import faiss
else:
    raise 'Operating system not supported: %s' % op_sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use GPU?
gpu = False


def main(args):
    true_dict = get_true_dict()
    logger.info('Start training')
    g = train(gan_model=args.gan_model, gpu=gpu, logger=logger)
    for k in [1, 5]:
        print('P@{} : {}'.format(k, get_precision_k(k, g, true_dict)))


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan-model', type=int, default=1,
                        help='GAN Model {1,2,3} (default=1)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    gpu = args.gpu and torch.cuda.is_available()
    logger = init_logger('GAN')
    logger.info('Get true dict')
    main(args)
