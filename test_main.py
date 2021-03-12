from __future__ import print_function
import os
import torch
import test_options
from test_Processor import Processor


def main():
    args = test_options.parser.parse_args()
    # fix random seed for stable results
    torch.manual_seed(args.seed)
    # set visible gpus
    args.gpu_ids = visible_gpu(args.gpus)
    # create folder
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
        
    out_dir = os.path.join('./ckpt/', args.dataset_name, 'eval')
    log_dir = os.path.join('./logs/', args.dataset_name, 'eval')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    processor = Processor(args)
    processor.processing()


def visible_gpu(gpus):
    """
        set visible gpu.
        can be a single id, or a list
        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpus)))
    return list(range(len(gpus)))


if __name__ == '__main__':
    main()
