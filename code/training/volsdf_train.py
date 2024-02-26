import os
import sys
import time
from tqdm import tqdm
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import pdb
from skimage import measure
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as matplt
from PIL import Image


import utils.general as utils
import utils.plots_flow as plt
from utils import rend_util

class VolSDFTrainRunner():
    def __init__(self,**kwargs):
        # torch.set_default_dtype(torch.float32)
        # torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.is_dist = kwargs['is_dist']
        self.world_size = kwargs['world_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.device = torch.device('cuda', index=kwargs['gpu_index'])
        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        self.is_only_plot = kwargs['is_only_plot'] if 'is_only_plot' in kwargs else False
        self.num_frames = self.conf.get_int('model.num_frames')
        self.exp_postfix = self.conf.get_string('train.exp_postfix',default=None)
        try:
            scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        except:
            scan_id = self.conf.get_string('dataset.scan_id', default='')
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        if not self.is_only_plot:
            self.timestamp = kwargs.get('timestamp_now', '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
            if self.exp_postfix is not None:
                self.timestamp = self.timestamp + self.exp_postfix
        else:
            self.timestamp = timestamp

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"
        if not self.is_only_plot:
            self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp, 'log')) if self.GPU_INDEX ==0 else None
            self.log_txt = os.path.join(self.expdir, self.timestamp, 'log', 'log.txt')
        else:
            self.writer = None
            self.log_txt = os.path.join(self.expdir, self.timestamp, 'log', 'log_plot.txt')

        if (self.GPU_INDEX == 0 or not self.is_dist) and (not self.is_only_plot):
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            utils.mkdir_ifnotexists(self.expdir)
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
            utils.mkdir_ifnotexists(self.plots_dir)
            utils.mkdir_ifnotexists(self.checkpoints_path)
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp, 'log'))
            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        if self.is_dist:
            dist.barrier()
        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        utils.log_print(self.log_txt, 'shell command : {0}'.format(' '.join(sys.argv)))

        utils.log_print(self.log_txt, 'Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = kwargs['train_dataset']
        self.ds_len = len(self.train_dataset)
        self.train_frames = self.train_dataset.training_frames
        self.train_num_frames = len(self.train_frames)
        utils.log_print(self.log_txt, 'Finish loading data. Data-set size: {0}'.format(self.ds_len))
        # if scan_id < 24 and scan_id > 0: # BlendedMVS, running for 200k iterations
        #     self.nepochs = max(int(200000 / self.ds_len), self.nepochs)
        utils.log_print(self.log_txt, 'RUNNING FOR {0}'.format(self.nepochs))

        self.train_dataloader = kwargs['dataloader']
        self.plot_dataloader = kwargs['plot_dataloader']

        conf_model = self.conf.get_config('model')
        conf_model.__setitem__('load_optical_flow', self.conf.get_config('dataset').get_bool('load_optical_flow',False))
        self.conf_model = conf_model
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        self.model.to(device=self.device)

        # load pretrained sdf flow
        if not self.conf.get_string('train.pretrained_dir') == 'none':
            utils.log_print(self.log_txt, f">>> load pretrained flow net from {self.conf.get_string('train.pretrained_dir')}")
            pre_trained_flownet = torch.load(self.conf.get_string('train.pretrained_dir'), map_location=self.device)
            # convert weight norm state dict to normal
            model_state = {}
            is_the_dict_the_whole_model = False
            for k,v in pre_trained_flownet['model_state_dict'].items():
                if 'weight_g' in k and k.replace('weight_g','weight') not in model_state.keys():
                    wv = pre_trained_flownet['model_state_dict'][k.replace('weight_g', 'weight_v')]
                    wv = F.normalize(wv,dim=1)
                    model_state[k.replace('weight_g','weight')] = v*wv
                elif 'weight_v' in k:
                    continue
                else:
                    model_state[k] = v
                if not is_the_dict_the_whole_model and 'implicit_network' in k:
                    is_the_dict_the_whole_model = True

            if is_the_dict_the_whole_model:
                tmp = {}
                for k, v in model_state.items():
                    if 'implicit_network' in k:
                        tmp[k.replace('implicit_network.', '')] = v
                model_state = tmp

            self.model.implicit_network.load_state_dict(model_state, strict=False)
        # self.model = torch.compile(self.model)
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            utils.log_print(self.log_txt, f'load model from {old_checkpnts_dir}/ModelParameters/{str(kwargs["checkpoint"])}.pth.')
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"), map_location=self.device)
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            self.start_epoch = saved_model_state['epoch']
            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"), map_location=self.device)
            try:
                self.optimizer.load_state_dict(data["optimizer_state_dict"])
            except:
                utils.log_print(self.log_txt,'did not load optimizer state')
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"), map_location=self.device)
            try:
                self.scheduler.load_state_dict(data["scheduler_state_dict"])
            except:
                utils.log_print(self.log_txt,'did not load scheduler state')
        if self.is_dist:
            self.model = DDP(self.model, device_ids=[self.GPU_INDEX],find_unused_parameters=True)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')

    def save_checkpoints(self, epoch):
        if self.is_dist:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.module.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.module.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        else:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))


    def get_surface_points_marching_cube(self, tt, resolution=60, num_points=1000,
                                         voxel_size=0.04, xmin=np.array([-0.5,-0.5,-0.5]),
                                         xmax=np.array([1.0,1.0,1.0])):
        # get surface points
        with torch.no_grad(), autocast(enabled=True):
            self.model.eval()
            # grid_boundary = [-3.0, 3.0]
            # grid = plt.get_grid_uniform(resolution, grid_boundary)
            grid = plt.get_grid_uniform_notcube(voxel_size, xmin, xmax)
            z = []
            step = self.conf_model.get_int('step')
            num_frames = self.conf_model.get_int('num_frames')
            points = grid['grid_points']

            if tt < self.num_frames-1:
                t = tt + 1
            else:
                t = tt
            t = torch.linspace(0, t, t*step+1,device=t.device)*2/(num_frames-1) - 1
            if self.is_dist:
                sdf = lambda x, t: self.model.module.implicit_network.get_outputs_tmp(x, t, step=step, ord=0)
            else:
                sdf = lambda x, t: self.model.implicit_network.get_outputs_tmp(x, t, step=step, ord=0)

            for pnts in torch.split(points, 100000, dim=0):
                z.append(sdf(pnts, t.unsqueeze(0))[1][:, :, :1].cpu())
            z = torch.cat(z, dim=0)
            z = z.float().data.numpy()

            # [t-step+1, ..., t-1, t, t+1, ..., t+step-1]
            meshs = []
            surface_points = []
            surface_norms = []
            if z.shape[1] == 1:
                if z[:,0].min() < 0 and z[:,0].max() > 0:
                    verts, faces, normals, values = measure.marching_cubes(
                        volume=z[:,0].reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                              grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=0,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][0][2] - grid['xyz'][0][1]))

                    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
                    vmin = np.min(verts,axis=0)
                    vmax = np.max(verts,axis=0)
                    xmin[vmin<xmin] = vmin[vmin<xmin]
                    xmin[vmax>vmax] = vmax[vmax>vmax]
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    mesh.compute_vertex_normals()
                    mesh.compute_triangle_normals()
                    pcd = mesh.sample_points_uniformly(num_points)
                    surface_point = np.array(pcd.points)
                    surface_norm = np.array(pcd.normals)
                    meshs.append(mesh)
                    surface_points.append(torch.from_numpy(surface_point).to(dtype=torch.get_default_dtype(),device=self.device))
                    surface_norms.append(torch.from_numpy(surface_norm).to(dtype=torch.get_default_dtype(),device=self.device))
                else:
                    # surface_point = np.zeros([1,3])
                    # surface_norm = np.zeros([1,3])
                    # mesh = None
                    meshs.append(None)
                    surface_points.append(None)
                    surface_norms.append(None)

                self.model.train()
                return surface_points, surface_norms, meshs, xmin, xmax

            for i in range(tt*step-step+1,tt*step+step):
                if i < 0:
                    meshs.append(None)
                    surface_points.append(None)
                    surface_norms.append(None)
                    continue
                elif i > (self.num_frames-1)*step:
                    meshs.append(None)
                    surface_points.append(None)
                    surface_norms.append(None)
                    continue

                if z[:,i].min() < 0 and z[:,i].max() > 0:
                    verts, faces, normals, values = measure.marching_cubes(
                        volume=z[:,i].reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                              grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=0,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][0][2] - grid['xyz'][0][1]))

                    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
                    vmin = np.min(verts,axis=0)
                    vmax = np.max(verts,axis=0)
                    xmin[vmin<xmin] = vmin[vmin<xmin]
                    xmin[vmax>vmax] = vmax[vmax>vmax]
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    mesh.compute_vertex_normals()
                    mesh.compute_triangle_normals()
                    pcd = mesh.sample_points_uniformly(num_points)
                    surface_point = np.array(pcd.points)
                    surface_norm = np.array(pcd.normals)
                    meshs.append(mesh)
                    surface_points.append(torch.from_numpy(surface_point).to(dtype=torch.get_default_dtype(),device=self.device))
                    surface_norms.append(torch.from_numpy(surface_norm).to(dtype=torch.get_default_dtype(),device=self.device))

                else:
                    # surface_point = np.zeros([1,3])
                    # surface_norm = np.zeros([1,3])
                    # mesh = None
                    meshs.append(None)
                    surface_points.append(None)
                    surface_norms.append(None)
                # # plot
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(surface_points)
                # o3d.visualization.draw_geometries([mesh,pcd])

            self.model.train()
            return surface_points, surface_norms, meshs,  xmin, xmax

    def run(self):
        utils.log_print(self.log_txt, "training...")
        train_split_pixels = self.conf.get_int('train.train_split_pixels', default=10000)
        scaler = GradScaler() # for mixture precision
        is_amp_t = 10
        xmin=np.array([-1.0,-1.0,-1.0])
        xmax = np.array([1.0, 1.0, 1.0])
        for epoch in range(self.start_epoch, self.nepochs + 1):
            st = time.time()
            if epoch % self.checkpoint_freq == 0 and (self.GPU_INDEX == 0 or not self.is_dist) and (not self.is_only_plot):
                self.save_checkpoints(epoch)
            if self.is_dist:
                dist.barrier()

            """plot"""
            if (self.do_vis and (epoch+1) % self.plot_freq == 0) or self.is_only_plot:
            # if True:
                render_image = False
                eval_img = False
                self.model.eval()
                with torch.no_grad(), autocast(enabled=True):
                    self.train_dataset.change_sampling_idx(-1)
                    if self.is_dist:
                        self.train_dataset.change_idx(-1)

                    """save rendered images and eval it"""
                    if eval_img:
                        psnrs = []
                        for idx in tqdm(range(len(self.train_dataset))):
                            img_key = self.train_dataset.img_keys[idx]
                            indices, model_input_tmp, ground_truth = self.train_dataset.__getitem__(idx)

                            model_input = {}
                            for k, v in model_input_tmp.items():
                                model_input[k] = v.to(device=self.device)[None]
                            split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                            res = []
                            for s in tqdm(split):
                                out = self.model(s)
                                d = {}
                                if 'rgb_values' in out and out['rgb_values'] is not None:
                                    rgb_values = out['rgb_values']
                                    d['rgb_values'] = rgb_values
                                else:
                                    d['rgb_values'] = torch.zeros([s['uv'].shape[1], 3], device=self.device)
                                if 'rgb_grid0' in out and out['rgb_grid0'] is not None:
                                    rgb_grid = out['rgb_grid0'][0]
                                    d['rgb_grid'] = rgb_grid
                                if "s_flow_pixel" in out and out['s_flow_pixel'] is not None:
                                    d['s_flow'] = out['s_flow_pixel']
                                d['normal_map'] = out['normal_map']

                                torch.cuda.empty_cache()
                                res.append(d)

                            batch_size = 1
                            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                            model_outputs['motion_mask'] = model_input['motion_mask']
                            psnr = rend_util.get_psnr(model_outputs['rgb_values'].reshape(-1, 3)[model_input['motion_mask'][0]==1],
                                                      ground_truth['rgb'].to(device=self.device).reshape(-1, 3)[model_input['motion_mask'][0]==1])
                            psnrs.append(psnr.item())
                            utils.log_print(self.log_txt,
                                            f'>>> data index {indices:03d} epoch {epoch:04d} psnr {psnr.item():.3f}')
                            img = model_outputs['rgb_values'].reshape([self.img_res[0],self.img_res[1],3]) * 255
                            img = img.cpu().data.numpy().astype(np.uint8)
                            img = Image.fromarray(img)
                            os.makedirs(f'{self.plots_dir}/rendered_img/',exist_ok=True)
                            img.save(f'{self.plots_dir}/rendered_img/{epoch}_{img_key}.jpg')
                        psnrs = np.array(psnrs)
                        mean = psnrs.mean()
                        txt = 'psnr\t'
                        for psnr in psnrs:
                            txt += str(psnr) + '\t'
                        txt += str(mean.item())
                        utils.log_print(self.log_txt, txt)

                        print("plot finished")
                        return

                    # self.train_dataset.change_idx(0)
                    indices, model_input_tmp, ground_truth = next(iter(self.plot_dataloader))
                    #
                    model_input = {}
                    for k,v in model_input_tmp.items():
                        model_input[k] = v.to(device=self.device)

                    if render_image:
                        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                        res = []
                        for s in tqdm(split):
                            out = self.model(s)
                            d = {}
                            if 'rgb_values' in out and out['rgb_values'] is not None:
                                rgb_values = out['rgb_values']
                                d['rgb_values'] = rgb_values
                            else:
                                d['rgb_values'] = torch.zeros([s['uv'].shape[1],3],device=self.device)
                            if 'rgb_grid0' in out and out['rgb_grid0'] is not None:
                                rgb_grid = out['rgb_grid0'][0]
                                d['rgb_grid'] = rgb_grid
                            if "s_flow_pixel" in out and out['s_flow_pixel'] is not None:
                                d['s_flow'] = out['s_flow_pixel']
                            d['normal_map'] = out['normal_map']

                            torch.cuda.empty_cache()
                            res.append(d)

                        batch_size = ground_truth['rgb'].shape[0]
                        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                        model_outputs['motion_mask'] = model_input['motion_mask']
                        psnr = rend_util.get_psnr(model_outputs['rgb_values'].reshape(-1, 3), ground_truth['rgb'].to(device=self.device).reshape(-1, 3))
                        utils.log_print(self.log_txt,f'>>> data index {indices.item():03d} epoch {epoch:04d} psnr {psnr.item():.3f}')
                        plot_data = self.get_plot_data(model_outputs, model_input['pose'], ground_truth['rgb'])

                    else:
                        plot_data = None

                    t = torch.linspace(0, int(self.conf.get_int('model.num_frames') - 1),
                                       int(self.conf.get_int('model.num_frames') - 1) * self.conf.get_int('model.step') + 1)\
                        * 2 / (self.conf.get_int('model.num_frames') - 1) - 1
                    t = t.to(device=model_input['t'].device)
                    t = t[:max(self.train_dataset.training_frames)* self.conf.get_int('model.step') + 1]
                    plt.plot(self.model,
                             indices,
                             plot_data,
                             self.plots_dir,
                             epoch,
                             self.img_res,
                             t,
                             self.GPU_INDEX == 0,
                             t_idx=indices.item(),
                             **self.plot_conf
                             )

                    # del model_outputs, plot_data, model_input
                    torch.cuda.empty_cache()
                self.model.train()
                self.train_dataset.change_sampling_idx(self.num_pixels)
                if self.is_only_plot:
                    print("plot finished")
                    return

            # make sure all processes load the same image
            if self.is_dist:
                sample_order = np.random.permutation(self.ds_len).tolist()
                utils.log_print(self.log_txt, f'epoch {epoch}, cuda{self.GPU_INDEX} {sample_order}')
                self.train_dataset.change_idx(sample_order.pop(0))
                dist.barrier()

            loss_all = {}
            num_samp = 0
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input_tmp = {}
                for k,v in model_input.items():
                    model_input_tmp[k] = v.to(device=self.device)
                bs = model_input_tmp['pose'].shape[0]

                uvs = torch.split(model_input_tmp["uv"], train_split_pixels, dim=1)
                motion_masks = torch.split(model_input_tmp["motion_mask"], train_split_pixels, dim=1)

                rgbs = torch.split(ground_truth['rgb'], train_split_pixels, dim=1)
                bg_masks = torch.split(model_input_tmp['bg_mask'], train_split_pixels, dim=1)
                if 'optical_flow_back' in model_input_tmp.keys():
                    optical_flow_back = torch.split(model_input_tmp['optical_flow_back'], train_split_pixels, dim=1)
                    optical_flow_forward = torch.split(model_input_tmp['optical_flow_forward'], train_split_pixels, dim=1)


                if (epoch+1) % self.plot_freq == 0:
                    surface_points, surface_norms, meshs,  _, _ = self.get_surface_points_marching_cube(model_input_tmp['t'][0],
                                                                                                        num_points=2000,
                                                                                                        xmin=xmin, xmax=xmax)
                    if ((epoch+1) % self.plot_freq == 0 or epoch == 0):
                        for mi, mesh in enumerate(meshs):
                            if mesh is not None:
                                o3d.io.write_triangle_mesh(f'{self.plots_dir}/mesh_intermedia_{epoch}_{model_input_tmp["t"][0].item()}_{mi:d}.ply', mesh)
                    torch.cuda.empty_cache()

                model_outputs_all = {}
                loss_all_tmp = {}
                self.optimizer.zero_grad()

                for ii, uv, gt_rgb, bg_mask in zip(list(range(len(uvs))), uvs, rgbs, bg_masks):
                    model_input_tmp['uv'] = uv
                    model_input_tmp['motion_mask'] = motion_masks[ii]
                    if 'optical_flow_back' in model_input_tmp.keys():
                        model_input_tmp['optical_flow_back'] = optical_flow_back[ii]
                        model_input_tmp['optical_flow_forward'] = optical_flow_forward[ii]

                    is_auto_cast = False

                    with autocast(enabled=is_auto_cast):
                        model_outputs = self.model(model_input_tmp, wvolume_consist=ii == (len(uvs) - 1),
                                                   volume_split=(self.GPU_INDEX, self.world_size))

                        gt = {'rgb':gt_rgb,'bg_mask':bg_mask}
                        if 'optical_flow_back' in model_input_tmp.keys():
                            gt['optical_flow_back'] = optical_flow_back[ii]
                            gt['optical_flow_forward'] = optical_flow_forward[ii]

                        loss_output = self.loss(model_outputs, gt, is_dist=self.is_dist, gpu=self.GPU_INDEX, world_size=self.world_size)

                        loss = loss_output['loss']
                        # update log infos
                        for k,v in model_outputs.items():
                            if k  == 'rgb_values':
                                value = model_outputs[k].detach() if model_outputs[k] is not None else \
                                    torch.zeros_like(gt['rgb'][0,:,None]).to(device=self.device)
                                if k not in model_outputs_all.keys():
                                    model_outputs_all[k] = value
                                else:
                                    model_outputs_all[k] = torch.cat([model_outputs_all[k], value],dim=0)
                            elif k == 'rgb_grid0' and model_outputs[k] is not None:
                                value = model_outputs[k].detach()#.transpose(1,0)
                                if k not in model_outputs_all.keys():
                                    model_outputs_all[k] = value
                                else:
                                    model_outputs_all[k] = torch.cat([model_outputs_all[k], value], dim=1)

                        for k,v in loss_output.items():
                            loss_all_tmp[k] = v.detach() + (0 if k not in loss_all_tmp.keys() else loss_all_tmp[k])

                        if is_auto_cast:
                            # self.optimizer.zero_grad()
                            scaler.scale(loss/len(uvs)).backward()
                        else:
                            (loss/len(uvs)).backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                if self.writer is not None:
                    self.writer.add_scalar('grad_norm', grad_norm.item(), epoch*self.ds_len + data_index+1)

                if is_auto_cast:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                psnr = rend_util.get_psnr(model_outputs_all['rgb_values'].reshape(-1,3) if 'rgb_values' in model_outputs_all and
                                                                                           model_outputs_all['rgb_values'] is not None
                                          else model_outputs_all['rgb_grid0'][0],
                                          ground_truth['rgb'].to(device=self.device).reshape(-1,3))
                for k,v in loss_all_tmp.items():
                    loss_all.update({k:v.item() + (0 if k not in loss_all.keys() else loss_all[k])})
                loss_all.update({'psnr':psnr.item()*len(uvs) + (0 if 'psnr' not in loss_all.keys() else loss_all['psnr'])})
                num_samp += len(uvs)
                self.scheduler.step()

                if self.is_dist and len(sample_order) > 0:
                    self.train_dataset.change_idx(sample_order.pop(0))
                if self.is_dist:
                    dist.barrier()

            log_dir = ""
            # also write the beta value
            if self.is_dist:
                beta = self.model.module.density.get_beta()#beta_min
            else:
                beta = self.model.density.get_beta()#.beta_min
            if self.writer is not None:
                self.writer.add_scalar('beta', beta, epoch)
                
            log_dir += f'beta={beta:.5f} '
            for k,v in loss_all.items():
                if self.writer is not None:
                    self.writer.add_scalar(k, v/num_samp, epoch)
                log_dir += f'{k}={v/num_samp:.5f} '
            utils.log_print(self.log_txt, f'cuda{self.GPU_INDEX:01d}_{indices[0].item():02d}_{self.expname}_{self.timestamp}_{epoch:05d}_{(time.time()-st):.3f}: {log_dir}')

            del loss_all, model_input, model_input_tmp, model_outputs, model_outputs_all
            torch.cuda.empty_cache()

        if (self.GPU_INDEX == 0 or not self.is_dist):
            self.save_checkpoints(epoch)

    def get_plot_data(self, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        rgb_grid = None
        if 'rgb_grid' in model_outputs:
            rgb_grid = model_outputs['rgb_grid'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        plot_data = {
            'rgb_gt': rgb_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
        }
        if rgb_grid is not None:
            plot_data['rgb_grid'] = rgb_grid

        return plot_data


    def generate_training_samples(self,num_sample,fn):
        """
        assume the ball grows from 0.5m to 1.0m
        """
        r = torch.linspace(0.5, 1.0, fn)[None, :, None]

        xyzs_off = torch.rand([num_sample, fn, 3])*6 - 3
        norm = xyzs_off.norm(dim=-1,keepdim=True)
        sdf_off = norm-r
        norm_off = xyzs_off/(norm+1e-10)

        xyzs_on = torch.randn([num_sample, fn, 3])
        norm = xyzs_on.norm(dim=-1, keepdim=True)
        xyzs_on[norm[:,:,0]<1e-5] = xyzs_on[norm[:,:,0]<1e-5] + 0.1
        norm = xyzs_on.norm(dim=-1, keepdim=True)
        xyzs_on = xyzs_on / norm
        norm_on = xyzs_on.clone()

        xyzs_on = xyzs_on * r

        sdf_on =  torch.zeros_like(xyzs_on[:,:,:1])

        return torch.cat([xyzs_on,xyzs_off],dim=0).cuda(), torch.cat([sdf_on,sdf_off],dim=0).cuda(), torch.cat([norm_on,norm_off],dim=0).cuda()

    def run_pretrain_flow(self):
        utils.log_print(self.log_txt, "training...")
        num_samples = self.conf.get_int('train.num_pixels', default=2000)
        scaler = GradScaler()  # for mixture precision
        is_amp_t = 10
        xmin = np.array([-1.0, -1.0, -1.0])
        xmax = np.array([1.0, 1.0, 1.0])
        for epoch in range(self.start_epoch, self.nepochs + 1):
            st = time.time()
            if epoch % self.checkpoint_freq == 0 and (self.GPU_INDEX == 0 or not self.is_dist) and (not self.is_only_plot):
                self.save_checkpoints(epoch)
            if self.is_dist:
                dist.barrier()

            t = torch.linspace(0, int(self.conf.get_int('model.num_frames') - 1),
                               int(self.conf.get_int('model.num_frames') - 1) * self.conf.get_int(
                                   'model.step') + 1) \
                * 2 / (self.conf.get_int('model.num_frames') - 1) - 1
            t = t.to(device=self.device)
            t = t[:self.num_frames * self.conf.get_int('model.step') + 1]

            """plot"""
            if (self.do_vis and (epoch+1) % self.plot_freq == 0) or self.is_only_plot:
                self.model.eval()
                with torch.no_grad(), autocast(enabled=True):

                    plot_data = None

                    t = torch.linspace(0, int(self.conf.get_int('model.num_frames') - 1),
                                       int(self.conf.get_int('model.num_frames') - 1) * self.conf.get_int('model.step') + 1)\
                        * 2 / (self.conf.get_int('model.num_frames') - 1) - 1
                    t = t.to(device=self.device)
                    plt.plot(self.model,
                             t[:1],
                             plot_data,
                             self.plots_dir,
                             epoch,
                             None,
                             t,
                             self.GPU_INDEX == 0,
                             t_idx=t[:1].item(),
                             **self.plot_conf
                             )

                    # del model_outputs, plot_data, model_input
                    torch.cuda.empty_cache()
                self.model.train()
                if self.is_only_plot:
                    print("plot finished")
                    return

            xyzs, sdf_gts, norm_gt = self.generate_training_samples(num_samples, self.num_frames)

            loss_all = {}
            self.optimizer.zero_grad()
            num_samp = 0
            fns = np.arange(0,self.num_frames)
            np.random.shuffle(fns)

            """plot"""
            if (self.do_vis and epoch % self.plot_freq == 0) or self.is_only_plot:
                surface_points, surface_norms, meshs, _, _ = self.get_surface_points_marching_cube(
                    torch.tensor(fns[0].item()).to(device=self.device),
                    num_points=2000,
                    xmin=xmin, xmax=xmax)
                # save intermediate meshes
                if (epoch % self.plot_freq == 0 or epoch == 0):
                    for mi, mesh in enumerate(meshs):
                        if mesh is not None:
                            o3d.io.write_triangle_mesh(
                                f'{self.plots_dir}/mesh_intermedia_{epoch}_{fns[0]}_{mi:d}.ply',
                                mesh)
                torch.cuda.empty_cache()

            for fn in fns:
                tt = t[:fn * self.conf.get_int('model.step') + 1][None]

                is_auto_cast = False

                with autocast(enabled=is_auto_cast):
                    out, gradients, flow = self.model.implicit_network(xyzs[:,fn], tt, ord=1)

                    sdf_est = out[:,-1, :1]
                    gradients = gradients[:, :, 0, :3].reshape([-1, 3])

                    loss_sdf = (sdf_est[:num_samples]-sdf_gts[:num_samples,fn]).abs().mean()
                    loss_inter = torch.exp(-1e2 * torch.abs(sdf_est[num_samples:])).mean()
                    grad_norm = torch.norm(gradients,dim=-1,keepdim=True)
                    loss_normal = torch.mean(((gradients[:num_samples]/(grad_norm[:num_samples]+1e-10) * norm_gt[:num_samples,fn]).sum(dim=-1) - 1)**2)
                    loss_grad = torch.abs(gradients.norm(dim=-1) - 1).mean()

                    loss = 3e3 * loss_sdf + 3e3 * loss_inter + 5e1 * loss_grad + 3e3 * loss_normal
                    if is_auto_cast:
                        scaler.scale(loss/self.num_frames).backward()
                    else:
                        (loss/self.num_frames).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            if is_auto_cast:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()

            loss_all['loss_sdf'] = loss_sdf.detach() + (0 if 'loss_grad' not in loss_all.keys() else loss_all['loss_sdf'])
            loss_all['loss_inter'] = loss_inter.detach() + (0 if 'loss_inter' not in loss_all.keys() else loss_all['loss_inter'])
            loss_all['loss_grad'] = loss_grad.detach() + (0 if 'loss_grad' not in loss_all.keys() else loss_all['loss_grad'])
            loss_all['loss_normal'] = loss_normal.detach() + (0 if 'loss_normal' not in loss_all.keys() else loss_all['loss_normal'])
            if self.writer is not None:
                self.writer.add_scalar('grad_norm', grad_norm.item(), epoch)

            self.scheduler.step()

            log_dir = ""
            for k, v in loss_all.items():
                if self.writer is not None:
                    self.writer.add_scalar(k, v / self.num_frames, epoch)
                log_dir += f'{k}={v / self.num_frames:.5f} '
            utils.log_print(self.log_txt,
                            f'cuda{self.GPU_INDEX:01d}_{self.expname}_{self.timestamp}_{epoch:05d}_{(time.time() - st):.3f}: {log_dir}')

            del loss_all
            torch.cuda.empty_cache()

        if (self.GPU_INDEX == 0 or not self.is_dist):
            self.save_checkpoints(epoch)

