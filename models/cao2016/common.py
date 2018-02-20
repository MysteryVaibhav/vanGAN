#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


def init_logger(name='logger'):
    """Initialize and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def generate_batch(items, batch_size):
    """Generate batches with a given size."""
    l = len(items)
    for pos in range(0, l, batch_size):
        yield items[pos:min(pos + batch_size, l)]
