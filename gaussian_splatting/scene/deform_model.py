import torch
from utils.time_utils import DeformNetwork
from utils.general_utils import get_expon_lr_func



class DeformModel:
    def __init__(self, is_2dgs=False):
        self.is_2dgs = is_2dgs
        self.deform = DeformNetwork(is_2dgs=is_2dgs).to("cuda:0")
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {
                "params": list(self.deform.parameters()),
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.training_args = training_args
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
        )
