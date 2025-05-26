import math
import torch
import numpy as np
import os
from oodgs.utils import read_input_data, get_git_root


class PoseSamplingBuilder:
    """This class is a collection of functions that sample a new pose from a given pose."""
    DEVICE = 'cpu'
    T_SIGMA = 15.0

    @staticmethod
    def sample_vector_near(stddev):
        """
        Samples a vector near [0,0,-1] with norm 1, where the distance from [0,0,-1] follows a Gaussian distribution.

        Args:
            stddev (float): Standard deviation of the Gaussian distribution for the distance.

        Returns:
            torch.Tensor: The sampled vector with norm 1.
        """
        v = torch.tensor([0.0, 0.0, -1.0], device=PoseSamplingBuilder.DEVICE)
        random_vector = torch.randn_like(v, device=PoseSamplingBuilder.DEVICE)

        # vector ortogonal
        orthogonal_vector = random_vector - torch.dot(random_vector, v) * v
        orthogonal_vector = orthogonal_vector / torch.norm(orthogonal_vector)

        distance = torch.randn(1, device=PoseSamplingBuilder.DEVICE) * stddev

        new_vector = (v + distance * orthogonal_vector).float()
        new_vector = new_vector / torch.norm(new_vector)
        # upper hemisphere
        new_vector[1] = torch.abs(new_vector[1])
        return new_vector

    @staticmethod
    def normalize(v):
        norm = torch.norm(v, dim=-1, keepdim=True)
        return v / norm

    @staticmethod
    def create_rotation_matrix(z_cam: torch.Tensor):
        # Reference up vector (world y-axis)
        v_ref = torch.tensor([0.0, 1.0, 0.0], device=PoseSamplingBuilder.DEVICE)
        
        # Check if z_cam is too aligned with v_ref
        dot_product = torch.dot(z_cam, v_ref)
        epsilon = 1e-6
        if torch.abs(dot_product) < 1 - epsilon:
            x_cam = PoseSamplingBuilder.normalize(torch.cross(v_ref, z_cam, dim=-1))
        else:
            # Use x-axis as fallback
            v_ref_alt = torch.tensor([1.0, 0.0, 0.0], device=PoseSamplingBuilder.DEVICE)
            x_cam = PoseSamplingBuilder.normalize(torch.cross(v_ref_alt, z_cam, dim=-1))
        
        y_cam = PoseSamplingBuilder.normalize(torch.cross(z_cam, x_cam, dim=-1))
        R = torch.stack([x_cam, y_cam, z_cam], dim=1).T
        return R

    @staticmethod
    def spherical_pose_from_eye(look_at_point: torch.Tensor, stddev: float):
        """
        Samples a new pose similar to the identity pose, where it is sampled from a sphere centered at the look_at_point
        and the radius is distance from the original pose to the look_at_point. The distance from the original pose
        follows a Gaussian distribution with standard deviation stddev. Each sampled pose is still looking 
        directly at the look_at_point (meaning that look_at_point is contained on the optical axis of the new pose camera).
        """
        pose = torch.eye(4, device=PoseSamplingBuilder.DEVICE)
        look_at_point = look_at_point.float()
        
        look_at_cam = pose @ torch.cat([look_at_point, torch.tensor([1.0], device=PoseSamplingBuilder.DEVICE)])
        d = look_at_cam[2]
        
        u = PoseSamplingBuilder.sample_vector_near(stddev)
        C_new = look_at_point + d * u
        
        z_cam = PoseSamplingBuilder.normalize(look_at_point - C_new)  # Direction from C_new to look_at_point
        R_new = PoseSamplingBuilder.create_rotation_matrix(z_cam)

        T_new = -R_new @ C_new
        
        M_new = torch.eye(4, device=PoseSamplingBuilder.DEVICE)
        M_new[:3, :3] = R_new
        M_new[:3, 3] = T_new
        return M_new
    
    @staticmethod
    def get_rotation_matrix(theta_rad: float, axis: str) -> torch.Tensor:
        if axis == 'x':
            return torch.tensor([
                [1, 0, 0],
                [0, math.cos(theta_rad), -math.sin(theta_rad)],
                [0, math.sin(theta_rad), math.cos(theta_rad)]
            ], device=PoseSamplingBuilder.DEVICE)
        elif axis == 'y':
            return torch.tensor([
                [math.cos(theta_rad), 0, math.sin(theta_rad)],
                [0, 1, 0],
                [-math.sin(theta_rad), 0, math.cos(theta_rad)]
            ], device=PoseSamplingBuilder.DEVICE)
        elif axis == 'z':
            return torch.tensor([
                [math.cos(theta_rad), -math.sin(theta_rad), 0],
                [math.sin(theta_rad), math.cos(theta_rad), 0],
                [0, 0, 1]
            ], device=PoseSamplingBuilder.DEVICE)
        else:
            raise ValueError(f'Invalid axis: {axis}')
    
    @staticmethod
    def rotate_pose_around_axis(pose: torch.Tensor, theta: float, axis: str) -> torch.Tensor:
        theta_rad = theta * np.pi / 180
        pose_inv = torch.inverse(pose)

        R = PoseSamplingBuilder.get_rotation_matrix(theta_rad, axis)
        rotation_matrix = R @ pose_inv[:3, :3]
        pose_inv[:3, :3] = rotation_matrix

        pose = torch.inverse(pose_inv)
        return pose
    
    @staticmethod
    def translate_pose(pose: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pose_inv = torch.inverse(pose)
        t_hom = torch.cat([t, torch.ones(1, device=pose.device)])
        t_hom_wcf = pose_inv @ t_hom
        t_wcf = t_hom_wcf[:3]
        new_pose_inv = pose_inv.clone()
        new_pose_inv[:3, 3] = t_wcf
        new_pose = torch.inverse(new_pose_inv)
        return new_pose
    
    @staticmethod
    def translate_along_circle(pose: torch.Tensor, depth_map: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        This implements the sampling of translation vector from https://arxiv.org/pdf/2502.07615.
        The idea is to first calculate the radius with respect to the mean depth, hyperparameter T_SIGMA
        and the focal length of the camera.
        """
        mean_focal_length = (K[0, 0] + K[1, 1]) / 2
        radius = PoseSamplingBuilder.T_SIGMA * torch.mean(depth_map) / mean_focal_length
        theta = torch.rand(1, device=PoseSamplingBuilder.DEVICE) * 2 * np.pi
        translation = torch.tensor([
            radius * torch.sin(theta),
            radius * torch.cos(theta),
            0
        ], device=PoseSamplingBuilder.DEVICE)
        new_pose = PoseSamplingBuilder.translate_pose(pose, translation)
        return new_pose
    
    @staticmethod
    def height_correction(pose: torch.Tensor, ref_pose: torch.Tensor) -> torch.Tensor:
        """
        This function corrects the height of the pose with respect to the reference pose.
        """
        pose_inv = torch.inverse(pose)
        ref_pose_inv = torch.inverse(ref_pose)
        pose_inv[2, 3] = ref_pose_inv[2, 3]
        return torch.inverse(pose_inv)
    
    @staticmethod
    def spherical_pose_augmentation(pose, sphere_radius: float, stddev: float):
        """
        Samples a new pose similar to the pose, where it is sampled from a sphere centered at the look_at_point
        and the radius is distance from the original pose to the look_at_point. The distance from the original pose
        follows a Gaussian distribution with standard deviation stddev. Each sampled pose is still looking 
        directly at the look_at_point (meaning that look_at_point is contained on the optical axis of the new pose camera).
        """
        pose = pose.to(PoseSamplingBuilder.DEVICE)
        look_at_point = torch.tensor([0.0, 0.0, sphere_radius], device=PoseSamplingBuilder.DEVICE)
        sampled_pose_ccf = PoseSamplingBuilder.spherical_pose_from_eye(look_at_point, stddev)
        sampled_pose_wcf = torch.inverse(pose) @ sampled_pose_ccf
        sampled_pose_wcf = torch.inverse(sampled_pose_wcf)
        return sampled_pose_wcf


class PoseSampler:
    DEVICE = 'cpu'
    
    def __init__(self, strategy: str, theta: float = None, sphere_radius: float = None, stddev: float = None):
        self.theta = theta
        self.sphere_radius = sphere_radius
        assert stddev > 0, f'stddev must be greater than 0, got {stddev}'
        self.stddev = stddev
        self.strategy = strategy

    def ood_translation_only(self, pose: torch.Tensor, depth_map: torch.Tensor, K: torch.Tensor, **kwargs):
        """
        Similar to sampling from flow distillation paper (https://arxiv.org/pdf/2502.07615.pdf).
        Takes in the pose, performs downward rotation by theta, and then translates along
        the circle with radius T_SIGMA * mean_depth / mean_focal_length.
        """
        pose = pose.to(PoseSampler.DEVICE)
        pose = PoseSamplingBuilder.rotate_pose_around_axis(pose, self.theta, 'x')
        pose = PoseSamplingBuilder.translate_along_circle(pose, depth_map, K)
        return pose
    
    def ood_spherical_augmentation(self, pose: torch.Tensor, **kwargs):
        """
        Performs the downward rotation by theta, and then samples a new pose from a sphere centered at the look_at_point.

        """
        pose = pose.to(PoseSampler.DEVICE)
        pose = PoseSamplingBuilder.rotate_pose_around_axis(pose, self.theta, 'x')
        pose = PoseSamplingBuilder.spherical_pose_augmentation(pose, self.sphere_radius, self.stddev)
        return pose
    
    def ood_spherical_auto(self, pose: torch.Tensor, depth_map: torch.Tensor, auto_height_correction: bool = True, **kwargs):
        """
        Performs translation forward followed by spherical augmentation. Estimates optimal parameters from depth map.
        
        Args:
            pose (torch.Tensor): The pose to augment.
            depth_map (torch.Tensor): The depth map from the pose.

        Returns:
            torch.Tensor: The augmented pose.
        """
        # param estimation
        translation_fwd_mean = depth_map.median() / 2
        translation_fwd_std = translation_fwd_mean * 0.1
        translation_fwd = torch.normal(mean=translation_fwd_mean, std=translation_fwd_std)
        
        translated_pose = PoseSamplingBuilder.translate_pose(pose, torch.tensor([0.0, 0.0, translation_fwd], device=PoseSampler.DEVICE))
        translated_pose = PoseSamplingBuilder.height_correction(translated_pose, pose) if auto_height_correction else translated_pose
        
        spherical_radius = depth_map.median() / 2
        out_pose = PoseSamplingBuilder.spherical_pose_augmentation(translated_pose, spherical_radius, self.stddev)
        return out_pose
    
    def uniform_driving(self, pose: torch.Tensor, depth_map: torch.Tensor, auto_height_correction: bool = True, **kwargs):
        """
        Performs translation forward followed by spherical augmentation. Estimates optimal parameters from depth map.
        
        Args:
            pose (torch.Tensor): The pose to augment.
            depth_map (torch.Tensor): The depth map from the pose.

        Returns:
            torch.Tensor: The augmented pose.
        """
        # param estimation
        translation_fwd_max = depth_map.median() / 2
        translation_fwd_min = - translation_fwd_max
        translation_fwd = torch.rand(1, device=PoseSampler.DEVICE) * (translation_fwd_max - translation_fwd_min) + translation_fwd_min
        
        translated_pose = PoseSamplingBuilder.translate_pose(pose, torch.tensor([0.0, 0.0, translation_fwd], device=PoseSampler.DEVICE))
        translated_pose = PoseSamplingBuilder.height_correction(translated_pose, pose) if auto_height_correction else translated_pose
        
        spherical_radius = depth_map.median() / 2
        out_pose = PoseSamplingBuilder.spherical_pose_augmentation(translated_pose, spherical_radius, self.stddev)
        return out_pose
    
    
    def object_centric_spherical(self, pose: torch.Tensor, depth_map: torch.Tensor, **kwargs):
        """
        This is a go-to strategy for object-centric scenes. Performs only the spherical pose augmentation.
        """
        pose = pose.to(PoseSampler.DEVICE)
        spherical_radius = depth_map.median() / 2
        out_pose = PoseSamplingBuilder.spherical_pose_augmentation(pose, spherical_radius, self.stddev)
        return out_pose
    
    def translate(self, pose):
        """
        Samples a new pose by translating along the circle with radius T_SIGMA * mean_depth / mean_focal_length.
        """
        pose = pose.to(PoseSampler.DEVICE)
        pose = PoseSamplingBuilder.translate_pose(pose, torch.tensor([0.0, 0.0, 5.0], device=PoseSampler.DEVICE))
        return pose

    def sample(self, pose, depth_map: torch.Tensor = None, K: torch.Tensor = None):
        """
        This function provides interface for sampling a new pose. Internally, it calls the appropriate method
        based on the strategy. Some arguments are used only for certain strategies.
        """
        if self.strategy == 'translation_only':
            return self.ood_translation_only(pose, depth_map, K)
        elif self.strategy == 'spherical_augmentation':
            return self.ood_spherical_augmentation(pose)
        elif self.strategy == 'spherical_auto':
            return self.ood_spherical_auto(pose, depth_map)
        elif self.strategy == 'object_centric_spherical':
            return self.object_centric_spherical(pose, depth_map)
        elif self.strategy == 'uniform_driving':
            return self.uniform_driving(pose, depth_map)
        else:
            raise ValueError(f'Invalid strategy: {self.strategy}')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='spherical_auto')
    parser.add_argument('--theta', type=float, default=10.0)
    parser.add_argument('--sphere_radius', type=float, default=1.0)
    parser.add_argument('--stddev', type=float, default=0.1)
    args = parser.parse_args()

    # Read input data from directory
    data_dir = os.path.join(get_git_root(), 'samples')
    train_rgb, ood_train, train_depth, ood_depth, train_pose, ood_pose, calibration_matrix = read_input_data(data_dir)
    
    # Use the first pose and depth map for testing
    pose = train_pose[0]  # shape: (4,4)
    depth_map = train_depth[0]  # shape: (H,W)
    
    pose_sampler = PoseSampler(args.strategy, args.theta, args.sphere_radius, args.stddev)
    pose = pose_sampler.sample(pose, depth_map, calibration_matrix)
    print(pose)
    
