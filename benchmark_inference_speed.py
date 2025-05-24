import os
import time
import argparse
import torch
from src.ood_loss import OODLoss
from src.utils import read_input_data, get_git_root
from torchvision import transforms


def benchmark_inference(loss_type: str, num_runs: int = 100,
                      image_size: int = None):
    ood_loss = OODLoss()
    
    data_dir = os.path.join(get_git_root(), 'samples')
    train_rgb, ood_train, train_depth, ood_depth, train_pose, ood_pose, \
        calibration_matrix = read_input_data(data_dir)
    
    if image_size is not None:
        # resizing is useful to see the effect of the image size on the inference time
        resize_transform = transforms.Resize((image_size, image_size))
        # (H, W, C) -> (C, H, W) -> (H, W, C)
        train_rgb = resize_transform(
            train_rgb.permute(2, 0, 1)).permute(1, 2, 0)
        ood_train = resize_transform(
            ood_train.permute(2, 0, 1)).permute(1, 2, 0)
        # (H, W) -> (1, H, W) -> (H, W)
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
    
    if loss_type == 'met3r':  # warmup so that caching doesn't affect results
        _ = ood_loss(train_rgb, ood_train, train_depth, ood_depth,
                    train_pose, ood_pose, calibration_matrix)
    else:
        _ = ood_loss.forward_depth_mvc(train_depth, ood_depth,
                                      train_pose, ood_pose, calibration_matrix)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        if loss_type == 'met3r':
            _ = ood_loss(train_rgb, ood_train, train_depth, ood_depth,
                        train_pose, ood_pose, calibration_matrix)
        else:
            _ = ood_loss.forward_depth_mvc(train_depth, ood_depth,
                                          train_pose, ood_pose, calibration_matrix)
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    print(f'Average inference time for {loss_type}: {avg_time*1000:.2f}ms')


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference time for OOD losses')
    parser.add_argument('loss', type=str, choices=['met3r', 'depth_mvc'],
                      help='Type of loss to benchmark')
    parser.add_argument('--runs', type=int, default=10,
                      help='Number of runs for benchmarking, results are averaged')
    parser.add_argument('--image-size', type=int, default=None,
                      help='Square image size to resize inputs to (e.g. 256 for 256x256)')
    
    args = parser.parse_args()
    benchmark_inference(args.loss, args.runs, args.image_size)


if __name__ == '__main__':
    main()
