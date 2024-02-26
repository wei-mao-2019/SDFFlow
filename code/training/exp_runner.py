import sys
# for cluster training
sys.path.append('/home/wei/Documents/projects/2022-nerf/SDFFlow/code')
import os
import argparse
import GPUtil
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pyhocon import ConfigFactory

from training.volsdf_train import VolSDFTrainRunner
import utils.general as utils

def main(gpu, args):
    if args.is_dist:
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(gpu)
    dtype = torch.float32
    # dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.cuda.set_device(gpu)
    # torch.set_float32_matmul_precision('high')

    conf = ConfigFactory.parse_file(args.conf)
    dataset_conf = conf.get_config('dataset')
    if args.scan_id != -1:
        dataset_conf['scan_id'] = args.scan_id

    train_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)
    train_dataset.change_sampling_idx(conf.get_int('train.num_pixels'))
    if args.is_dist:
        data_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                       num_replicas=args.world_size,
                                                                       rank=rank)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=train_dataset.collate_fn,
                                num_workers=0, pin_memory=True)

        plot_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=train_dataset.collate_fn,
                                    num_workers=0, pin_memory=True, sampler=data_sampler)
    else:
        dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       collate_fn=train_dataset.collate_fn
                                                       )
        plot_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=conf.get_int('plot.plot_nimgs'),
                                                      shuffle=True,
                                                      collate_fn=train_dataset.collate_fn
                                                      )

    trainrunner = VolSDFTrainRunner(conf=args.conf,
                                    batch_size=args.batch_size,
                                    nepochs=args.nepoch,
                                    expname=args.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=args.exps_folder,
                                    is_continue=args.is_continue,
                                    timestamp=args.timestamp,
                                    checkpoint=args.checkpoint,
                                    scan_id=args.scan_id,
                                    do_vis=not args.cancel_vis,
                                    train_dataset=train_dataset,
                                    dataloader=dataloader,
                                    plot_dataloader=plot_dataloader,
                                    timestamp_now=args.timestamp_now,
                                    is_dist=args.is_dist,
                                    world_size=args.world_size,
                                    is_only_plot=args.is_only_plot
                                    )

    if args.is_pretrain_flow:
        trainrunner.run_pretrain_flow()
    else:
        trainrunner.run()
    if args.is_dist:
        dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--nepoch', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument('--is_only_plot', default=False, action="store_true",
                        help='If set, only plot results donot create new repo.')
    parser.add_argument('--is_pretrain_flow', default=False, action="store_true",
                        help='If set, only plot results donot create new repo.')

    parser.add_argument('--is_dist', action='store_true', default=False)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    args.timestamp_now = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

    if args.is_dist:
        print(args.gpus * args.nodes)
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8887'
        print(1)
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        if args.gpu == "auto":
            deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                            excludeID=[], excludeUUID=[])
            gpu = deviceIDs[0]
        else:
            gpu = int(args.gpu)
        main(gpu, args)
