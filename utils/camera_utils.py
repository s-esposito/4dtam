import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.slam_utils import image_gradient, image_gradient_mask
import torch.nn.functional as F
from utils.normal_utils import intrins_to_intrins_inv, get_cam_coords, d2n_tblr


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        fid=None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.fid = fid
        self.device = device

        self.T = torch.eye(4, device=device).to(torch.float32)
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone()

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

        intrins = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.intrins_inv = intrins_to_intrins_inv(intrins).float().unsqueeze(0).to(0)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]

        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
            fid=float(idx) / len(dataset),
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform  # TODO: Need to invert for high order SHs by inverse_t(self.world_view_transform).

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1, keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:], size, stride=size).squeeze(0)
            self.grad_mask = I_f
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = 0.05
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(
            *mask_shape
        )
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask

        self.rgb_pixel_mask = rgb_pixel_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask

        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(
                dtype=torch.float32, device=self.device
            )[None]

            depth = self.gt_depth.unsqueeze(0)
            points = get_cam_coords(self.intrins_inv, depth)
            normal, valid_mask = d2n_tblr(points, d_min=1e-3, d_max=1000.0)
            normal = normal * valid_mask
            self.normal = normal
            self.normal_raw = self.normal.squeeze(0).permute(1, 2, 0).cpu().numpy()
            self.mask = valid_mask

        if self.mask is not None:
            self.rgb_pixel_mask = self.rgb_pixel_mask * self.mask
            self.rgb_pixel_mask_mapping = self.rgb_pixel_mask_mapping

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None
        self.normal = None
        self.normal_raw = None


class CameraMsg:
    def __init__(self, Camera=None, uid=None, T=None, T_gt=None, fid=None):
        if Camera is not None:
            self.uid = Camera.uid
            self.T = Camera.T
            self.T_gt = Camera.T_gt
            self.fid = Camera.fid
        else:
            self.uid = uid
            self.T = T
            self.T_gt = T_gt
            self.fid = fid
