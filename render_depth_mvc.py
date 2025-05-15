import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from src.ood_loss import OODLoss
from src.utils import read_input_data, get_git_root


ood_loss = OODLoss()
# Read input data from directory
data_dir = os.path.join(get_git_root(), 'samples')

# Load RGB images, depth maps, poses and calibrationx
train_rgb, ood_train, \
    train_depth, ood_depth, \
    train_pose, ood_pose, \
    calibration_matrix = read_input_data(data_dir)

# Compute depth multi-view consistency (MVC) score map
score, score_map, oclusion_mask = ood_loss.forward_depth_mvc(
    gt_depth=train_depth,
    ood_depth=ood_depth,
    gt_pose=train_pose,
    ood_pose=ood_pose,
    calibration_matrix=calibration_matrix,
    use_oclusion_mask=True
)

# save score_map as heatmap
score_map = score_map.cpu().numpy().squeeze()  # Remove singleton dimension (H,W,1) -> (H,W)
plt.figure(figsize=(10, 8))
plt.imshow(score_map, cmap='viridis')
plt.colorbar(label='Depth MVC L1 Error')
plt.title('Depth Multi-View Consistency (MVC) Score Heatmap')
plt.axis('off')
plt.savefig(os.path.join(data_dir, 'depth_mvc_score_map_heatmap.png'), bbox_inches='tight', dpi=300)
plt.close()
