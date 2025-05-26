import os
import argparse
import torch
from oodgs.ood_loss import OODLoss
from oodgs.utils import read_input_data, get_git_root
from torchvision import transforms


def benchmark_vram(loss_type: str, image_size: int = None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ood_loss = OODLoss()
    
    data_dir = os.path.join(get_git_root(), 'samples')
    train_rgb, ood_train, train_depth, ood_depth, train_pose, ood_pose, \
        calibration_matrix = read_input_data(data_dir)
    
    if image_size is not None:
        resize_transform = transforms.Resize((image_size, image_size))
        train_rgb = resize_transform(
            train_rgb.permute(2, 0, 1)).permute(1, 2, 0)
        ood_train = resize_transform(
            ood_train.permute(2, 0, 1)).permute(1, 2, 0)
        train_depth = resize_transform(
            train_depth.unsqueeze(0)).squeeze(0)
        ood_depth = resize_transform(
            ood_depth.unsqueeze(0)).squeeze(0)
        
        scale_factor = image_size / train_rgb.shape[0]
        calibration_matrix = calibration_matrix.clone()
        calibration_matrix[0, 0] *= scale_factor  # fx
        calibration_matrix[1, 1] *= scale_factor  # fy
        calibration_matrix[0, 2] *= scale_factor  # cx
        calibration_matrix[1, 2] *= scale_factor  # cy
    
    if loss_type == 'met3r':
        _ = ood_loss(train_rgb, ood_train, train_depth, ood_depth,
                    train_pose, ood_pose, calibration_matrix)
    else:
        _ = ood_loss.forward_depth_mvc(train_depth, ood_depth,
                                      train_pose, ood_pose, calibration_matrix)
    
    print(f'Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB')


def main():
    parser = argparse.ArgumentParser(description='Benchmark VRAM usage for OOD losses')
    parser.add_argument('loss', type=str, choices=['met3r', 'depth_mvc'],
                      help='Type of loss to benchmark')
    parser.add_argument('--image-size', type=int, default=None,
                      help='Square image size to resize inputs to (e.g. 256 for 256x256)')
    
    args = parser.parse_args()
    benchmark_vram(args.loss, args.image_size)


if __name__ == '__main__':
    main() 