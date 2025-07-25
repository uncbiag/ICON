#python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=152.2.132.95 --master_port=1232 unigradicon_train_parallel.py                                         



from datetime import datetime
from icon_registration.config import device
import icon_registration.unicarl.fixed_point_carl as fpc
import footsteps
import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchvision.utils
import os
import icon_registration.unicarl
import icon_registration.unicarl.unigradicon_train_parallel
os.environ["OMP_NUM_THREADS"]="8"

if __name__ == "__main__":

    import datasets
    from datasets import input_shape

    net = icon_registration.unicarl.unigradicon_train_parallel.make_net(3, input_shape, False)

    BATCH_SIZE = 4

    icon_registration.unicarl.unigradicon_train_parallel.train_batchfunction(
        net,
        icon_registration.unicarl.unigradicon_train_parallel.make_make_pair(datasets),
        steps=450000
    )

