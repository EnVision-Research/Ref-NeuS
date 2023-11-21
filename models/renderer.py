import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import open3d as o3d
from models.utils import project, normalize


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    ref_idx,
                    uv,
                    dataset,
                    inter_mesh,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        
        # Parameters for projection
        inv_c2w_all = dataset.inv_pose_all.cuda()
        intrinsics_all = dataset.intrinsics_all.cuda()
        scr_ind = [i for i in range(inv_c2w_all.shape[0])]
        scr_ind.remove(ref_idx)

        inv_src_pose = inv_c2w_all[scr_ind]
        src_intr = intrinsics_all[scr_ind][:, :3, :3]

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        normals = F.normalize(gradients, dim=-1)
        
        refdirs = 2.0 * torch.sum(normals * -dirs, axis=-1, keepdims=True) * normals + dirs
        sampled_color = color_network(pts, gradients, refdirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        
        # Normal map
        normals_map = F.normalize(gradients.reshape(batch_size, n_samples, 3), dim=-1)
        normals_map = (normals_map * weights[:, :128, None]).sum(dim=-2).detach().cpu().numpy()
        
        # Reflection Score
        RS = 10. * torch.ones(batch_size, dtype=torch.float32).cuda()
        if inter_mesh is not None:
            # __________________________________________________________________________________________________________
            # _______________________________ Localize surface points with predicted sdf _______________________________
            # __________________________________________________________________________________________________________
            sdf_d = sdf.reshape(batch_size, n_samples)
            prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
            sign = prev_sdf * next_sdf

            surf_exit_inds = torch.unique(torch.where(sign < 0)[0])

            sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
            idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
            tmp = torch.einsum("ab,b->ab", (sign, idx))
            prev_idx = torch.argmax(tmp, 1, keepdim=True)
            next_idx = prev_idx + 1

            prev_inside_sphere = torch.gather(inside_sphere, 1, prev_idx)
            next_inside_sphere = torch.gather(inside_sphere, 1, next_idx)
            mid_inside_sphere = (0.5 * (prev_inside_sphere + next_inside_sphere) > 0.5).float()

            sdf1 = torch.gather(sdf_d, 1, prev_idx)
            sdf2 = torch.gather(sdf_d, 1, next_idx)
            z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
            z_vals2 = torch.gather(mid_z_vals, 1, next_idx)
            
            z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
            z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
            max_z_val = torch.max(z_vals)
            z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
            points_for_warp = (rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]).detach()

            # __________________________________________________________________________________________________________
            # _________________________________________ Occlusion Detection ____________________________________________
            # __________________________________________________________________________________________________________        
            ref_point_dir = torch.cat((rays_o, rays_d), dim=-1).cpu().numpy()
            ref_point_dir = o3d.core.Tensor(ref_point_dir, dtype=o3d.core.Dtype.Float32)

            ans_ref = inter_mesh.cast_rays(ref_point_dir)
            t_hit_ref = torch.from_numpy(ans_ref['t_hit'].numpy()).cuda().squeeze(0)
            
            # inf means the ray dose not hit the surface
            val_ray_inds = torch.where(~torch.isinf(t_hit_ref))[0]
            tmp = list(set(val_ray_inds.cpu().numpy()) & set(surf_exit_inds.cpu().numpy())) # double check: sdf network and mesh
            val_ray_inside_inds = torch.tensor(tmp).cuda()
            
            if val_ray_inside_inds.shape[0] != 0:
                
                # get source rays_o 
                rays_o_src = dataset.all_rays_o.cuda()[scr_ind]
                # get all rays_d for all validate point 
                rays_d_scr = points_for_warp - rays_o_src
                rays_d_scr = F.normalize(rays_d_scr, dim=-1)
               
                rays_o_src = rays_o_src.expand(rays_d_scr.size())

                points_for_warp = points_for_warp[val_ray_inside_inds]

                #cal source
                val_rays_o_scr = rays_o_src[val_ray_inside_inds]
                val_rays_d_scr = rays_d_scr[val_ray_inside_inds]

                all_point_dir = torch.cat((val_rays_o_scr, val_rays_d_scr),dim=-1).cpu().numpy()
                all_point_dir = o3d.core.Tensor(all_point_dir, dtype=o3d.core.Dtype.Float32)
                ans_source = inter_mesh.cast_rays(all_point_dir)
                
                t_hit_src = torch.from_numpy(ans_source['t_hit'].numpy()).cuda()
                t_hit_src[torch.where(torch.isinf(t_hit_src))] = -10.

                # distance from surface points to source rays_o 
                dist = ((points_for_warp.repeat(1, len(scr_ind), 1) - val_rays_o_scr) / val_rays_d_scr)[..., 0]
                # we slightly relax the occlusion judegment. If the surfaces are optimized inward, all source views are occluded. 
                dist_ref = ((points_for_warp.squeeze(1) - rays_o[val_ray_inside_inds]) / rays_o[val_ray_inside_inds])[..., 0].detach()
                diff_ref = (dist_ref - t_hit_ref[val_ray_inside_inds]).detach()
            
                diff_ref[torch.where(torch.isinf(diff_ref))] = 0.
                diff_ref[torch.where(diff_ref < 0 )] = 0.
    
                val_inds = torch.where(( (dist - 1.5 * diff_ref[:,None].repeat(1, rays_d_scr.shape[1]) - 0.05) <= t_hit_src))
                all_val_inds = torch.zeros(val_rays_d_scr.shape[:2], dtype=torch.int64).cuda()
                all_val_inds[val_inds] = 1
               
                with torch.no_grad():
                    grid_px, in_front = project(points_for_warp.view(-1, 3), inv_src_pose[:, :3].cuda(), src_intr[:, :3, :3].cuda())
                    grid_px[..., 0], grid_px[..., 1] = grid_px[..., 1].clone(), grid_px[..., 0].clone()

                    grid = normalize(grid_px.squeeze(0), dataset.H, dataset.W, clamp=10)
                    warping_mask_full = (in_front.squeeze(0) & (grid < 1).all(dim=-1) & (grid > -1).all(dim=-1))

                    sampled_rgb_vals = F.grid_sample(dataset.images[scr_ind].squeeze(0).permute(0, 3, 2, 1), grid.unsqueeze(1), align_corners=True).squeeze(2).transpose(1, 2)
                    sampled_rgb_vals[~warping_mask_full, :] = 0  #[num_scr, num_val_rays, 3]
                    all_rgbs_warp = sampled_rgb_vals.transpose(0, 1) #[num_val_rays, num_scr, 3]

                    bk_ind = torch.all(all_rgbs_warp == 0, dim=2) 
                    all_val_inds_fina = all_val_inds * warping_mask_full.transpose(0,1) * ~bk_ind
                    
                    num_val = torch.sum(all_val_inds_fina, dim=-1) 
                    bk_num = torch.sum(bk_ind * all_val_inds * warping_mask_full.transpose(0,  1), dim=-1)
                    _val_ind = torch.where((num_val>=5) & (bk_num <= 10))
        
                    # here we use L1 distance, which achieves similar results
                    RS_temp = 10. * torch.ones((val_ray_inside_inds.shape[0])).cuda()
                    uv_val = uv[val_ray_inside_inds]
    
                    anchor_rgb = dataset.images[ref_idx][(uv_val[:,0].long(), uv_val[:,1].long())].view(-1, 3).cuda()
                    diff_color = torch.zeros_like(all_rgbs_warp).cuda()
                    all_warp_color = torch.zeros_like(all_rgbs_warp).cuda()
                    
                    diff_color[all_val_inds_fina.bool()] = torch.abs(all_rgbs_warp - anchor_rgb[:, None, :].expand(all_rgbs_warp.size()))[all_val_inds_fina.bool()]
                    all_warp_color[all_val_inds_fina.bool()] = torch.abs(all_rgbs_warp - anchor_rgb[:, None, :].expand(all_rgbs_warp.size()))[all_val_inds_fina.bool()]
                    val_mean = torch.sum(all_warp_color, dim=-2) / num_val.unsqueeze(-1)
                        
                    RS_temp[_val_ind] = (val_mean.sum(-1)[_val_ind] * 10.).clamp(min=1., max=5.)
        
                    RS[val_ray_inside_inds] = RS_temp
                    
        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'RS': RS,
            'normal_map': normals_map,
        }

    def render(self, rays_o, rays_d, near, far, img_idx, uv, dataset, inter_mesh, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    img_idx,
                                    uv,
                                    dataset,
                                    inter_mesh,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'RS': ret_fine['RS'],
            'normal_map': ret_fine['normal_map'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
