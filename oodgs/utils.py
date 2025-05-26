import PIL
import torch
import os
import numpy as np

import os
from git import Repo, InvalidGitRepositoryError

def get_git_root(path=None):
    """
    Returns the root directory of the git repository containing the given path.
    If no path is provided, uses the current file's directory.
    """
    if path is None:
        path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

    while dir_path != os.path.dirname(dir_path):  # Stop at filesystem root
        try:
            repo = Repo(dir_path, search_parent_directories=True)
            return repo.git.rev_parse("--show-toplevel")
        except InvalidGitRepositoryError:
            dir_path = os.path.dirname(dir_path)

    raise FileNotFoundError("No Git repository found in any parent directory.")

def read_input_data(data_dir: str) -> torch.Tensor:
    """
    Read necessary data to calculate the OOD loss. This includes the train and ood rgb images,
    train and ood depth images, train and ood poses and calibration matrix. This function assumes they all reside in the same directory.
    """
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    assert os.path.exists(os.path.join(data_dir, 'train_rgb.png')), f"Train RGB image {os.path.join(data_dir, 'train_rgb.png')} does not exist"
    assert os.path.exists(os.path.join(data_dir, 'ood_train.png')), f"OOD RGB image {os.path.join(data_dir, 'ood_train.png')} does not exist"

    assert os.path.exists(os.path.join(data_dir, 'train_depth.npy')), f"Train depth image {os.path.join(data_dir, 'train_depth.npy')} does not exist"
    assert os.path.exists(os.path.join(data_dir, 'ood_depth.npy')), f"OOD depth image {os.path.join(data_dir, 'ood_depth.npy')} does not exist"
    
    assert os.path.exists(os.path.join(data_dir, 'train_pose.npy')), f"Train pose {os.path.join(data_dir, 'train_pose.npy')} does not exist"
    assert os.path.exists(os.path.join(data_dir, 'ood_pose.npy')), f"OOD pose {os.path.join(data_dir, 'ood_pose.npy')} does not exist"
    
    assert os.path.exists(os.path.join(data_dir, 'calibration_matrix.npy')), f"Calibration matrix {os.path.join(data_dir, 'calibration_matrix.npy')} does not exist"

    train_rgb = PIL.Image.open(os.path.join(data_dir, 'train_rgb.png'))
    train_rgb = torch.tensor(np.array(train_rgb)).cuda()
    ood_train = PIL.Image.open(os.path.join(data_dir, 'ood_train.png'))
    ood_train = torch.tensor(np.array(ood_train)).cuda()

    train_depth = np.load(os.path.join(data_dir, 'train_depth.npy'))
    train_depth = torch.tensor(train_depth).cuda()
    ood_depth = np.load(os.path.join(data_dir, 'ood_depth.npy'))
    ood_depth = torch.tensor(ood_depth).cuda()

    train_pose = np.load(os.path.join(data_dir, 'train_pose.npy'))
    train_pose = torch.tensor(train_pose).cuda()
    ood_pose = np.load(os.path.join(data_dir, 'ood_pose.npy'))
    ood_pose = torch.tensor(ood_pose).cuda()
    
    calibration_matrix = np.load(os.path.join(data_dir, 'calibration_matrix.npy'))
    calibration_matrix = torch.tensor(calibration_matrix).cuda()
    
    return train_rgb, ood_train, train_depth, ood_depth, train_pose, ood_pose, calibration_matrix