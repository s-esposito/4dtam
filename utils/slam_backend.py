import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss
from gaussian_splatting.scene.deform_model import DeformModel

from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
import numpy as np

from utils.pose_utils import quat2rotmat_batch
from pytorch3d.ops import ball_query


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.is_2dgs = config["Training"]["4dtam"]["is_2dgs"]
        self.is_deform = config["Training"]["4dtam"]["is_deform"]

        self.use_normal = config["Training"]["4dtam"]["use_normal"]
        self.add_rigidity = config["Training"]["4dtam"]["add_rigidity"]

        self.finalized = False
        self.init_deform = False

    def set_deformation(self):
        if self.is_deform:
            self.warpfield = DeformModel(is_2dgs=self.is_2dgs)
            self.warpfield.train_setting(self.opt_params)

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        return

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        d_xyz, d_rotation, d_scaling = None, None, None
        reg_loss = 0
        itr_num = self.init_itr_num * 2 if self.is_deform else self.init_itr_num
        for mapping_iteration in range(itr_num):
            self.iteration_count += 1

            if self.is_deform and mapping_iteration > self.init_itr_num:
                time_input = (
                    torch.tensor([viewpoint.fid], device="cuda:0")
                    .unsqueeze(0)
                    .expand(self.gaussians.get_xyz.shape[0], -1)
                )
                d_xyz, d_rotation, d_scaling = self.warpfield.step(
                    self.gaussians.get_xyz.detach(), time_input
                )
                reg_loss = d_xyz.norm(dim=1).mean() + d_scaling.norm(dim=1).mean()

            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                is_2dgs=self.is_2dgs,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
            )

            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )

            loss_init += 0.1 * reg_loss
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if self.is_deform:
                    self.warpfield.optimizer.step()
                    self.warpfield.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def comput_rigidity_loss(
        self,
        prev_gaussian,
        cur_gaussian,
        prev_rotmat,
        cur_rotmat,
        neighbor_indices,
        neighbor_weight,
        prev_offset,
    ):
        return

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]

        if self.add_rigidity:
            radius = self.config["Training"]["4dtam"]["arap"]["radius"]
            num_knn = self.config["Training"]["4dtam"]["arap"]["num_knn"]
            subsample_num = self.config["Training"]["4dtam"]["arap"]["subsample_num"]
            weight_exp = self.config["Training"]["4dtam"]["arap"]["weight_exp"]
            arap_weight = self.config["Training"]["4dtam"]["arap"]["weight"]

        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)

        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for itr in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            d_xyz, d_rotation, d_scaling = None, None, None

            for cam_idx in range(len(current_window) - 1, -1, -1):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                if self.is_deform:
                    time_input = (
                        torch.tensor([viewpoint.fid], device="cuda:0")
                        .unsqueeze(0)
                        .expand(self.gaussians.get_xyz.shape[0], -1)
                    )
                    d_xyz, d_rotation, d_scaling = self.warpfield.step(
                        self.gaussians.get_xyz.detach(), time_input
                    )

                    if self.add_rigidity:
                        if itr == 0:
                            if cam_idx == len(current_window) - 1:  ##oldest keyframe
                                prev_gaussian = self.gaussians.get_xyz + d_xyz
                                subsample_idx = torch.randint(
                                    0, self.gaussians.get_xyz.shape[0], (subsample_num,)
                                )
                                knn_res = ball_query(
                                    prev_gaussian[subsample_idx][None],
                                    prev_gaussian[None],
                                    K=num_knn,
                                    radius=radius,
                                    return_nn=True,
                                )
                                neighbor_dist, neighbor_indices = (
                                    torch.sqrt(knn_res.dists),
                                    knn_res.idx,
                                )
                                neighbor_weight = (
                                    torch.exp(-weight_exp * knn_res.dists)
                                ) * (neighbor_indices > 0)
                                prev_offset = knn_res.knn[0] - prev_gaussian[
                                    subsample_idx
                                ].unsqueeze(1)

                                prev_gaussian_rot = (
                                    self.gaussians.get_rotation + d_rotation
                                )
                                prev_rotmat_full = quat2rotmat_batch(prev_gaussian_rot)
                                prev_rotmat = prev_rotmat_full[subsample_idx]

                                prev_rotmat_neighbor = prev_rotmat_full[
                                    neighbor_indices[0]
                                ]
                                prev_normal_diff = torch.sum(
                                    1
                                    - prev_rotmat[:, :, 2].unsqueeze(1)
                                    * (prev_rotmat_neighbor[:, :, :, 2]),
                                    dim=2,
                                )

                            elif cam_idx == 0:
                                cur_gaussian = self.gaussians.get_xyz + d_xyz
                                cur_gaussian_rot = (
                                    self.gaussians.get_rotation + d_rotation
                                )
                                sub = cur_gaussian[neighbor_indices[0]]

                                cur_offset = sub - cur_gaussian[
                                    subsample_idx
                                ].unsqueeze(1)

                                prev_rotmat_t = prev_rotmat.transpose(1, 2)
                                current_rotmat_full = quat2rotmat_batch(
                                    cur_gaussian_rot
                                )
                                current_rotmat = current_rotmat_full[[subsample_idx]]

                                relative_rotmat = torch.bmm(
                                    current_rotmat, prev_rotmat_t
                                )
                                cur_offset = cur_offset.permute(0, 2, 1)
                                transformed = torch.bmm(relative_rotmat, cur_offset)
                                transformed = transformed.permute(0, 2, 1)
                                rigidity_loss = torch.sqrt(
                                    ((prev_offset - transformed) ** 2).sum(-1)
                                    * neighbor_weight
                                    + 1e-20
                                ).mean()

                                current_rotmat_neighbor = current_rotmat_full[
                                    neighbor_indices[0]
                                ]
                                cur_normal_diff = torch.sum(
                                    1
                                    - current_rotmat[:, :, 2].unsqueeze(1)
                                    * (current_rotmat_neighbor[:, :, :, 2]),
                                    dim=2,
                                )
                                rigidity_normal_loss = torch.abs(
                                    (prev_normal_diff - cur_normal_diff)
                                    * neighbor_weight
                                    + 1e-20
                                ).mean()

                                loss_mapping += (
                                    arap_weight * rigidity_loss + rigidity_normal_loss
                                )

                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    is_2dgs=self.is_2dgs,
                    d_xyz=d_xyz,
                    d_rotation=d_rotation,
                    d_scaling=d_scaling,
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]

                if self.is_deform:
                    time_input = (
                        torch.tensor([viewpoint.fid], device="cuda:0")
                        .unsqueeze(0)
                        .expand(self.gaussians.get_xyz.shape[0], -1)
                    )
                    d_xyz, d_rotation, d_scaling = self.warpfield.step(
                        self.gaussians.get_xyz.detach(), time_input
                    )

                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    is_2dgs=self.is_2dgs,
                    d_xyz=d_xyz,
                    d_rotation=d_rotation,
                    d_scaling=d_scaling,
                )

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            if self.is_deform:
                scaling = self.gaussians.get_scaling + d_scaling
            else:
                scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += (
                10 * isotropic_loss.mean()
                if self.is_deform
                else 0.1 * isotropic_loss.mean()
            )

            if (
                self.use_normal
                and self.is_2dgs
                and viewpoint.normal is not None
                and itr == 0
            ):
                rend_normal = render_pkg["rend_normal"]
                depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
                gt_normal = viewpoint.normal
                gt_normal = (viewpoint.T[0:3, 0:3].T @ gt_normal.view(3, -1)).view(
                    image.shape[0], image.shape[1], image.shape[2]
                )
                normal_mask = gt_normal > 0
                normal_error = (
                    1
                    - (rend_normal * gt_normal * depth_pixel_mask * normal_mask).sum(
                        dim=0
                    )
                )[None].mean()
                loss_mapping += (
                    self.config["Training"]["4dtam"]["lr"]["normal_loss"] * normal_error
                )

            loss_mapping.backward()

            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                if self.is_deform:
                    self.warpfield.optimizer.step()
                    self.warpfield.optimizer.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

        return gaussian_split

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.T.clone()))

        if tag is None:
            tag = "sync_backend"

        msg = [
            tag,
            clone_obj(self.gaussians),
            self.occ_aware_visibility,
            keyframes,
            self.warpfield.deform.state_dict() if self.is_deform else None,
        ]

        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if not self.init_deform:
                self.set_deformation()
                self.init_deform = True

            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.001)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.001)
                    continue

                if self.single_thread:
                    time.sleep(0.001)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    if not self.finalized:
                        self.color_refinement()
                        self.finalized = True
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    current_window = current_window
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 5
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue

                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 10000
        err_array = {key: 0.0 for key in self.viewpoints.keys()}
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]

            d_xyz, d_rotation, d_scaling = None, None, None
            if self.is_deform:
                time_input = (
                    torch.tensor([viewpoint_cam.fid], device="cuda:0")
                    .unsqueeze(0)
                    .expand(self.gaussians.get_xyz.shape[0], -1)
                )
                d_xyz, d_rotation, d_scaling = self.warpfield.step(
                    self.gaussians.get_xyz.detach(), time_input
                )

            render_pkg = render(
                viewpoint_cam,
                self.gaussians,
                self.pipeline_params,
                self.background,
                is_2dgs=self.is_2dgs,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
                surf=self.is_2dgs,
            )

            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            gt_depth = viewpoint_cam.gt_depth
            gt_depth_mask = gt_depth > 0.0
            Ll1_depth = l1_loss(depth * gt_depth_mask, gt_depth * gt_depth_mask)
            scaling = (
                self.gaussians.get_scaling + d_scaling
                if self.is_deform
                else self.gaussians.get_scaling
            )
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss = (
                Ll1
                + 0.01 * Ll1_depth
                + 0.1 * isotropic_loss.mean()
                + 0.001 * d_scaling.mean()
            )

            if self.use_normal and self.is_2dgs:
                rend_normal = render_pkg["rend_normal"]
                surf_normal = render_pkg["surf_normal"]
                normal_error = (1 - (rend_normal * (-surf_normal)).sum(dim=0))[None]
                normal_loss = 0.005 * normal_error.mean()
                loss += normal_loss

            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

            if self.is_deform:
                err_array[viewpoint_cam_idx] = Ll1.item()
                if iteration % 20 == 0:
                    max_error_indices = np.argsort(err_array)[-2:]
                    viewpoint_idx_stack += max_error_indices.tolist()

        Log("Map refinement done")
