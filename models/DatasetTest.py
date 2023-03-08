"""
    Prepare the dataset for testing. We use only one object for testing.
    This is an example for the case of phase+attenuation training, for the case of attenuation only, please remove the phase part.
"""

from random import randint
import numpy as np
import torch
from models.mat_calc import get_world_mat
from models.utils import data_reg, data_preproc, new_reg


class TestDataset(torch.utils.data.Dataset):
    """Load data"""

    def __init__(self, opt):
        super().__init__()
        self.n_views = opt.n_views
        self.use_time = opt.use_time
        self.test_index = opt.test_obj_idx

        path = opt.load_path

        # Load images
        images = np.load(
            path
        )  # For ph+att: [dataset_size, num_projections, 2, H, W], otherwise [dataset_size, num_projections, H, W]
        att = data_preproc(images) 
        att = new_reg(att)
        images = np.concatenate((att, att), axis=2)

        if images.ndim == 4:
            self.images_pool = torch.from_numpy(images)[:, :, None, ...]
        elif images.ndim == 5:
            self.images_pool = torch.from_numpy(images)
        else:
            raise NotImplementedError(
                f"Please check the dataloader for supported dataset shape."
            )
        self.generate_poses()
        if self.use_time:
            self.generate_timestamp()

    def generate_poses(self):
        ProjAngles = np.linspace(
            0.0, 23.86 * 8, 9
        )  # NB: Modify this part to match the experimental setup

        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        world_mat = []
        for i in range(ProjAngles.shape[0]):
            world_mat.append(get_world_mat(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        self.tform_cam2world = torch.from_numpy(world_matrix).float()  # [100,4,4]

    def generate_timestamp(self):
        """Generate time stamps for the simulation dataset. Assuming that the objects change every second"""
        self.images_time = torch.arange(self.images_pool.shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.obj_idx = (
            self.test_index
        )  # We use a fixed validation object in each epoch. Change to random index if needed.
        self.imgs = self.images_pool[self.obj_idx]
        self.poses = self.tform_cam2world
        if self.use_time:
            self.times = self.images_time[self.obj_idx]
            return self.imgs, self.poses, self.times, self.obj_idx
        else:
            return self.imgs, self.poses, self.obj_idx
