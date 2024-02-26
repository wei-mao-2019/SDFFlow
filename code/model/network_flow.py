import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parameter import Parameter
from TorchDiffEqPack.odesolver import odesolve

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler, ErrorBoundSampler_WTime, UniformSampler, HierarchicalSampler_WTime
from model.module import *
from model.module import _parse_activation, first_layer_sine_init,sine_init
import utils.general as utils
# from siren import *

class MLP(nn.Module):
    def __init__(self, layers=[], activations=[], skips=[], weight_norm=False):
        '''
            n_dim: dimensionality of input/output spatial coordinates
            z_dim: dimensionality of hidden code
            h_dims: list of dimensions of intermediate layers
            activations: list of string ids for non-linear activations
        '''
        super().__init__()
        n_dim = layers[0]
        h_dims = layers[1:]
        assert len(h_dims) == len(activations)
        self.layers = nn.ModuleList()
        self.activations = list(map(_parse_activation, activations))
        self.n_dim = layers[0]
        self.skips = skips
        for i in range(len(h_dims)):
            out_dim = h_dims[i]
            # self.layers.append(torch.nn.Linear(n_dim+self.n_dim if i in skips else n_dim, out_dim))
            lin = nn.Linear(n_dim+self.n_dim if i in skips else n_dim, out_dim)
            if activations[i] == 'siren':
                if i == 0:
                    first_layer_sine_init(lin)
                else:
                    sine_init(lin)
            else:
                if i == len(h_dims) - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(n_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -0.6)
                elif i == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif (i+1) in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(self.n_dim - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            self.layers.append(lin)

            n_dim = out_dim

        self.out_dim = (h_dims[-1] if len(h_dims) > 0 else n_dim)

    def forward(self, x, dx_dh=None, d2x_dh2=None, ord=1):
        '''
            x: initial spatial coordinates of shape [n_points, n_dim]
            t: constant scaler for time variable for time-varying velocity fields,
               leave it as zero if velocity field should be time-constant
            z: optional latent vector of shape [z_dim] that the velocity field is conditioned on
            dx_dh: initial Jacobian
            d2x_dh2: initial Hessian diagonals
        '''
        x0 = x
        dx0_dh = dx_dh
        d2x0_dh2 = d2x_dh2
        i=0
        for linear, activation in zip(self.layers, self.activations):

            # dx_dh = _append_zero_rows(dx_dh, non_x_dims)
            # d2x_dh2 = _append_zero_rows(d2x_dh2, non_x_dims)

            if i in self.skips:
                x = torch.cat([x0,x],dim=-1)
                if dx_dh is not None:
                    dx_dh = torch.cat([dx0_dh,dx_dh],dim=1)
                if dx_dh is not None and ord > 1:
                    d2x_dh2 = torch.cat([d2x0_dh2,d2x_dh2],dim=1)

            x, dx_dh, d2x_dh2 = linear_2nd_ord(linear, x, dx_dh, d2x_dh2,ord=ord)
            x, dx_dh, d2x_dh2 = activation_2nd_ord(activation, x, dx_dh, d2x_dh2,ord=ord)

            i+=1
        return x, dx_dh, d2x_dh2


class ImplicitNetwork_RK2(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            activations,
            skip_in=(),
            weight_norm=True,
            multires=0,
            multires_t=0,
            sphere_scale=1.0,
            is_pos_emb=True,
            **kargs
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        self.d_in = d_in
        self.is_pos_emb = is_pos_emb
        dims = [d_in] + dims + [d_out + feature_vector_size]

        if is_pos_emb:
            self.emb_t = PosEmb(1, max_freq=multires_t - 1, num_freqs=multires_t)
            self.emb = PosEmb(d_in, max_freq=multires - 1, num_freqs=multires)
            self.mlp_initial = MLP(layers=[self.emb.out_dim] + dims[1:],
                                   activations=activations, skips=skip_in, weight_norm=weight_norm)
            self.mlp_flow = MLP(layers=[self.emb.out_dim + self.emb_t.out_dim] + dims[1:],
                                activations=activations, skips=skip_in, weight_norm=weight_norm)
        else:
            self.mlp_initial = MLP(layers=dims,
                                   activations=activations, skips=skip_in)
            self.mlp_flow = MLP(layers=[dims[0] + 1] + dims[1:],
                                activations=activations, skips=skip_in)

    def forward(self, input, t, step=3, ord=1, wscene_flow=False):
        """
        :param input: [b,3]
        :param t: [1,fn]
        :param step: scalar
        :param ord: {0,1,2}
        :return:
        """
        bs = input.shape[0]
        fn = t.shape[1]
        step_size = t[0,1] - t[0,0] if fn > 1 else 0

        out0 = self.forward_init(input, ord)
        out_last = out0
        out = [out0[:, None]]
        flow = []
        for i in range(fn):
            dout_t = self.forward_sdf(input, t[:, i:i+1]+step_size/2, ord=ord)
            flow.append(dout_t[:,None])
            if i < fn - 1:
                out_last = out_last + dout_t*step_size
                out.append(out_last[:,None])

        out = torch.cat(out, dim=1).reshape([bs, fn, self.mlp_flow.out_dim, -1])  # [1,47,257,5]
        flow = torch.cat(flow, dim=1).reshape([bs, fn, self.mlp_flow.out_dim, -1]) # [bs, t, 1]
        gradients = out[:, :, :, 1:]  # [bs, t, f, 4] if ord == 1
        out = out[:, :, :, 0]  # [bs, t, f]

        return out, gradients, flow

    def get_outputs(self, x, t, step=3, ord=1):
        # x.requires_grad_(True)
        # output, flow, dx_dh, d2x_dh2, scene_flow, dsf_dh_all, d2sf_dh2_all = self.forward(x,t,step,ord)
        output, gradient, flow = self.forward(x, t, step, ord)
        sdf = output[:, -1, :1]
        feature_vectors = output[:, -1, 1:]
        dx_dh = gradient #[bs, t, f, 4] if ord == 1
        # return sdf, flow, feature_vectors, dx_dh
        return sdf, flow, feature_vectors, dx_dh

    def get_outputs_2(self, x, t, step=3, ord=1):
        # x.requires_grad_(True)
        output, gradient, flow = self.forward(x, t, step, ord)
        sdf = output[:, -1, :1]
        feature_vectors = output[:, :, 1:]
        # return sdf, output[:, :, :1], flow, feature_vectors, dx_dh
        return sdf, output[:, :, :1], flow, feature_vectors, gradient

    def get_outputs_tmp(self, x, t, step=3, ord=1):
        # x.requires_grad_(True)
        # output, flow, dx_dh, d2x_dh2, scene_flow, dsf_dh_all, d2sf_dh2_all = self.forward(x,t,step,ord)
        output, gradient, flow = self.forward(x, t, step, ord)
        sdf = output[:, :, :1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf.unsqueeze(1))
        feature_vectors = output[:, :, 1:]
        # return sdf, output[:, :, :1], flow, feature_vectors, dx_dh
        return sdf, output[:, :, :1], flow, feature_vectors, gradient

    def get_sdf_vals(self, x, t, step=3, ord=0):
        # output, _, dx_dh, d2x_dh2, scene_flow, dsf_dh_all, d2sf_dh2_all = self.forward(x,t,step,ord)
        output, _, _ = self.forward(x, t, step, ord)
        sdf = output[:, -1, :1]
        return sdf

    def process_input(self, input, t=None, ord=0):

        dx_dh_xyz = None
        d2x_dh2_xyz = None
        if ord > 0:
            dx_dh_xyz = torch.eye(input.shape[-1], dtype=input.dtype, device=input.device)
            if ord > 1:
                d2x_dh2_xyz = torch.zeros(input.shape[-1], input.shape[-1], dtype=input.dtype, device=input.device)
        if t is not None:
            dx_dh_t = None
            d2x_dh2_t = None
            if ord > 0:
                dx_dh_t = torch.eye(1, dtype=input.dtype, device=input.device)
                if ord > 1:
                    d2x_dh2_t = torch.zeros(1, 1, dtype=input.dtype, device=input.device)
        if self.is_pos_emb:
            out_xyz, dx_dh_xyz, d2x_dh2_xyz = self.emb(input, dx_dh=dx_dh_xyz, d2x_dh2=d2x_dh2_xyz,
                                                       ord=ord)  # [bs,f], [bs,f,3], [bs,f,3]
            if t is not None:
                out_t, dx_dh_t, d2x_dh2_t = self.emb_t(t.reshape([-1, 1]), dx_dh=dx_dh_t, d2x_dh2=d2x_dh2_t,
                                                       ord=ord)  # [1,f], [1,f,1]
        else:
            out_xyz = input
            if t is not None:
                out_t = t
                if not (out_t.shape[0] == out_xyz.shape[0]):
                    out_t = out_t.repeat(out_xyz.shape[0], 1)
                    if ord > 0:
                        dx_dh_t = dx_dh_t.repeat(out_xyz.shape[0], 1, 1)
                        if ord > 1:
                            d2x_dh2_t = d2x_dh2_t.repeat(out_xyz.shape[0], 1, 1)
        if t is not None:
            if out_xyz.shape[0] != out_t.shape[0]:
                out_t = out_t.repeat(out_xyz.shape[0], 1)
            out = torch.cat([out_xyz, out_t], dim=-1)  # [bs,ft+fxyz]
            bs, nout = out_xyz.shape
            ninp = input.shape[1]
            dx_dh = None
            d2x_dh2 = None
            if ord > 0:
                dx_dh = torch.zeros([bs, out.shape[1], ninp + 1], dtype=input.dtype, device=input.device)
                dx_dh[:, :nout, :ninp] = dx_dh_xyz
                dx_dh[:, nout:, ninp:] = dx_dh_t
                if ord > 1:
                    d2x_dh2 = torch.zeros([bs, nout, ninp + 1], dtype=input.dtype, device=input.device)
                    d2x_dh2[:, :nout, :ninp] = d2x_dh2_xyz
                    d2x_dh2[:, nout:, ninp:] = d2x_dh2_t
        else:
            out = out_xyz
            dx_dh = dx_dh_xyz
            d2x_dh2 = d2x_dh2_xyz
        return out, dx_dh, d2x_dh2

    def forward_init(self, input, ord=0):
        input, dx_dh, d2x_dh2 = self.process_input(input, ord=ord)
        out0, dx_dh0, d2x_dh20 = self.mlp_initial(input, dx_dh=dx_dh, d2x_dh2=d2x_dh2, ord=ord)
        bs = out0.shape[0]
        if ord > 0:
            # add t dim for consistancy
            dx_dh0 = torch.cat([dx_dh0, torch.zeros_like(dx_dh0[:, :, :1])], dim=-1)

            out0 = torch.cat([out0[..., None], dx_dh0], dim=-1)
            if ord > 1:
                d2x_dh20 = torch.cat([d2x_dh20, torch.zeros_like(d2x_dh20[:, :, :1])], dim=-1)
                out0 = torch.cat([out0[..., None], d2x_dh20], dim=-1)
        return out0.reshape([bs, -1])

    def forward_sdf(self, input, t, ord=0):
        """
        :param input: [b,3]
        :param t: [1,1]
        :return:
        """
        input, dx_dh, d2x_dh2 = self.process_input(input, t, ord)
        bs = input.shape[0]
        dout_t, dx_dh, d2x_dh2 = self.mlp_flow(input, dx_dh=dx_dh, d2x_dh2=d2x_dh2, ord=ord)  # [bs, f], [bs, f, 4]
        if ord > 0:
            dout_t = torch.cat([dout_t[..., None], dx_dh], dim=-1)
            if ord > 1:
                dout_t = torch.cat([dout_t[..., None], d2x_dh2], dim=-1)
        return dout_t.reshape([bs, -1])


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            # embedview_fn, input_ch = get_embedder(multires_view)
            # self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            self.embedview_fn = PosEmb(3, max_freq=multires_view-1, num_freqs=multires_view)
            input_ch = self.embedview_fn.out_dim
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, t, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs,_,_ = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            if t.shape[0] == 1:
                rendering_input = torch.cat([points, t.repeat([points.shape[0],1]), view_dirs,
                                             normals, feature_vectors], dim=-1)
            else:
                rendering_input = torch.cat([points, t, view_dirs,
                                             normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


class VolSDFNetwork_RK2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.load_optical_flow = conf.get_bool('load_optical_flow', default=False)
        self.render_depth = conf.get_bool('render_depth', default=False)
        self.render_optical_flow = conf.get_bool('render_optical_flow', default=True)
        self.render_rgb = conf.get_bool('render_rgb', default=False)
        self.render_normal = conf.get_bool('render_normal', default=False)
        self.return_weights = conf.get_bool('return_weights', default=False)

        self.ode_method = conf.get_string('ode_method', default='none')
        self.num_frames = conf.get_float('num_frames', default=20.0)
        self.step = conf.get_int('step', default=3)
        self.scene_flow_type = conf.get_string('scene_flow_type', default='gt')
        self.sampler_type = conf.get_string('sampler_type', default='uniform')

        self.implicit_network = ImplicitNetwork_RK2(self.feature_vector_size,
                                                    0.0 if self.white_bkgd else self.scene_bounding_sphere,
                                                    **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        if self.sampler_type == 'uniform':
            self.ray_sampler = UniformSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        elif self.sampler_type == 'errorbound':
            self.ray_sampler = ErrorBoundSampler_WTime(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        elif self.sampler_type == 'hierarchical':
            self.ray_sampler = HierarchicalSampler_WTime(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def forward(self, input, ord=1, wscene_flow=None, wvolume_consist=True, volume_split=(0, 1)):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        if self.num_frames > 1:
            tt = torch.linspace(0, int(self.num_frames - 1), int(self.num_frames - 1) * self.step + 1,
                                device=input['t'].device) * 2 / (self.num_frames - 1) - 1
            step_size = 2 / (self.step * (self.num_frames - 1))
        else:
            tt = torch.tensor([-1], device=input['t'].device, dtype=intrinsics.dtype)
            step_size = 0
        tt = tt.unsqueeze(0)
        fi = input['t'][0]  # start from 0
        t = tt[:, :fi * self.step + 1]
        nt = t.shape[1]

        """get cam ray"""
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        """sample points"""
        if self.sampler_type == 'uniform':
            z_vals = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        else:
            z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, t[0], self)
        # get rid of nan depth values, for unknown reason sometimes there will be nan values
        if torch.isnan(z_vals).any():
            z_vals[torch.isnan(z_vals)] = torch.rand_like(z_vals[torch.isnan(z_vals)]) * self.scene_bounding_sphere * 2
            z_vals, _ = torch.sort(z_vals, -1)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        dirs_flat = dirs.reshape(-1, 3)

        """obtain sdf_flow, scene_flow and geometry feature"""
        out, gradients, flow = self.implicit_network(points_flat, t, ord=ord) # the flow is flow at t+0.5dt

        sdf_all = out[:,:,:1] # [t, bs, 1]
        feature_vectors_all = out[:,:,1:] # [t, bs, f-1]

        weights = self.volume_rendering(z_vals, sdf_all[:,-1])

        """volume rendering"""
        rgb_values = None
        if self.render_rgb:
            rgb_flat = self.rendering_network(points_flat, t[:, -1:], torch.zeros_like(gradients[:, -1, 0, :3]),
                                              torch.zeros_like(dirs_flat), feature_vectors_all[:, -1])
            rgb = rgb_flat.reshape(-1, N_samples, 3)
            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        if self.render_depth:
            depth = torch.sum(weights * z_vals, 1)

        normals = None
        if self.render_normal:
            normals = gradients[:, -1, 0, :3].reshape(-1,N_samples,3)
            normals = F.normalize(normals,dim=-1)
            normals = (weights.unsqueeze(-1)*normals).sum(dim=1).reshape([-1,N_uv_extra+1,3])

        output = {
            'rgb_values': rgb_values,
            'sdf_all': sdf_all[:, -1].reshape(-1, N_samples) if not self.return_weights else None,
            'depth': depth if self.render_depth else None,
            'normals': normals,
            'weights': weights if self.return_weights else None,
        }

        if self.training:
            output['grad_theta'] = gradients[:, :, 0, :3].reshape([-1, 3])

        if not self.training and gradients is not None:
            gradients = gradients.detach()[:, -1, 0, :3]
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]],
                                        dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights


