#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from tqdm import tqdm
import sys

counter = defaultdict(int)
for line in tqdm(sys.stdin):
    for w in line.strip().split():
        counter[w] += 1

for w, v in sorted(counter.items(), key=lambda t: -t[1]):
    print('{}\t{}'.format(w, v))
