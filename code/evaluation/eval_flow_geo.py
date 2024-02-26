import sys
sys.path.append('/home/wei/Documents/projects/2022-nerf/SDFFlow/code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from tqdm import tqdm

import utils.general as utils
import open3d as o3d

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    data_dir = conf.get_string('dataset.data_dir', '')
    scan_id = conf.get_string('dataset.scan_id', '')
    instance_dir = os.path.join('/home/wei/Documents/projects/2022-nerf/SDFFlow/data', data_dir,
                                     f'{scan_id}')

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != 'none' else conf.get_string('dataset.scan_id', default='none')
    if scan_id != 'none':
        expname = expname + '_{0}'.format(scan_id)
    else:
        scan_id = conf.get_string('dataset.object', default='')


    timestamp = kwargs['timestamp']

    expdir = os.path.join('/home/wei/Documents/projects/2022-nerf/SDFFlow/', exps_folder_name, expname)

    log_txt = os.path.join(expdir, timestamp, 'log', 'log_plot.txt')
    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    ckpt = kwargs['checkpoint']

    model_conf = conf.get_config('model')
    num_frames = model_conf.get_int('num_frames',24)
    train_frames = conf.get_list('dataset.training_frames', np.arange(num_frames).tolist())
    num_frames = max(train_frames)+1
    step = model_conf.get_int('step',2)
    cam_data = np.load(f'{instance_dir}/cameras.npz')
    scale_mat = torch.from_numpy(cam_data['scale_mat_0'].astype(np.float32)).cuda()
    accs = []
    comps = []
    for i in tqdm(range(num_frames)):
        mesh_est = o3d.io.read_triangle_mesh(f'{expdir}/{timestamp}/plots/surface_{ckpt:d}_{i*step:d}.ply')
        mesh_est.transform(scale_mat.cpu().data.numpy())
        o3d.io.write_triangle_mesh(f'{expdir}/{timestamp}/plots/scaled_surface_{ckpt:d}_{i*step:d}.ply', mesh_est)
        pcd_es = mesh_est.sample_points_uniformly(10000)
        points_est = torch.from_numpy(np.array(pcd_es.points)).float().cuda()
        # points_est = scale_mat[0,0] * points_est + scale_mat[:3,-1][None]
        if data_dir == 'our_synthesis_data':
            mesh_gt = o3d.io.read_triangle_mesh(f'{instance_dir}/mesh/scaled_remeshed_{i+1:03d}.ply')
            pcd_gt = mesh_gt.sample_points_uniformly(10000)
        elif data_dir == 'total_capture':
            mesh_gt = o3d.io.read_point_cloud(f'{instance_dir}/mesh/{i+1:04d}.ply')
            pcd_gt = mesh_gt.voxel_down_sample(voxel_size=1)

        points_gt = torch.from_numpy(np.array(pcd_gt.points)).float().cuda()/100
        pdist = torch.cdist(points_gt[None], points_est[None])[0]
        comp = torch.mean(torch.min(pdist,dim=1)[0]).cpu().data.numpy()*1000
        acc = torch.mean(torch.min(pdist,dim=0)[0]).cpu().data.numpy()*1000
        comps.append(comp)
        accs.append(acc)

    utils.log_print(log_txt, '')
    utils.log_print(log_txt, f'>>>>> evaluating exp {expdir}/{timestamp} checkpoint {ckpt}')
    txt = ''
    for tt in accs:
        txt += str(tt) + '\t'
    txt += str(np.mean(np.array(accs)).item())
    utils.log_print(log_txt,f'accuracy\t{txt}')
    txt = ''
    for tt in comps:
        txt += str(tt) + '\t'
    txt += str(np.mean(np.array(comps)).item())
    utils.log_print(log_txt,f'compeletness \t{txt}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=str, default='')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='2023_04_15_23_11_12', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='1650',type=int,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=str, default='none', help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=100, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--eval_mesh', default=False, action="store_true",
                        help='If set, save meshes.')
    parser.add_argument('--eval_rendering', default=False, action="store_true",
                        help='If set, evaluate rendering quality.')
    parser.add_argument('--texture_rendering', default=False, action="store_true",
                        help='If set, get mesh texture.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = int(opt.gpu)

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name=opt.evals_folder,
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             texture_rendering=opt.texture_rendering,
             eval_mesh=opt.eval_mesh,
             )
