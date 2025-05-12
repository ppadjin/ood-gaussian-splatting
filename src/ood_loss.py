import torch
import math
from submodules.met3r.met3r.met3r import MEt3R
from torchvision import transforms
from torch.nn import functional as F

class OODLoss(torch.nn.Module):
    # max allowed dimension for the image, important for fitting in VRAM of 24gb VRAM GPU
    MAX_ALLOWED_TOTAL_DIM = 640*640 
    DEVICE = 'cuda'        
    backbone_to_patch_size = {
        'dino16': 32,
        'dinov2': 14
    }
    showed_warning_padding = False
    
    def __init__(self, feat_backbone: str = 'dino16'):
        super().__init__()

        self.orig_size = None
        self.feat_backbone = feat_backbone
        self.patch_size = self.backbone_to_patch_size[feat_backbone]
        
        # by default, we don't pad the image
        self.pad = False
        self.metric = MEt3R(
            img_size=None,
            use_norm=True,
            feat_backbone=feat_backbone,
            patch_size=self.patch_size,
            featup_weights='mhamilton723/FeatUp',
            dust3r_weights='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
            use_mast3r_dust3r=True
        ).cuda()

    def __set_scaling_factor(self, image_size: tuple[int, int]):
        """
        Get the scaling factor for the image size. Allows to customize the scaling factor for different image sizes,
        but also provides the automatic scaling factor that will fit the image in VRAM.
        """
        if image_size[0] == 540 and image_size[1] == 960:
            self.scaling_factor = 8 / 15
        
        elif image_size[0] == 720 and image_size[1] == 960:
            self.scaling_factor = 2/3
        
        elif image_size[0] == 1706 and image_size[1] == 960: # iphone 14 pro
            self.scaling_factor = 1/3
        # in future if needed, provide more scaling factor for different sizes

        elif image_size[0] == 1080 and image_size[1] == 1920:  # full HD
            self.scaling_factor = 1/3
        else:
            # automatically determine a a scaling factor that will fit the image in VRAM
            current_total_dim = image_size[0] * image_size[1]
            self.scaling_factor = math.sqrt(Met3rLoss.MAX_ALLOWED_TOTAL_DIM / current_total_dim)

        if int(image_size[0] * self.scaling_factor) % self.patch_size != 0 or int(image_size[1] * self.scaling_factor) % self.patch_size != 0:
            if not self.showed_warning_padding:
                print(f"Warning: The image size {image_size} is not divisible by {self.patch_size}, padding will be applied")
                self.showed_warning_padding = True
            self.pad = True
            
    def preprocess_tensor(self, image: torch.Tensor, resize: bool = True) -> torch.Tensor:
        """
        Preprocess a tensor for MEt3R.
        Normalize to [-1,1] and resize if needed, using the scaling factor.
        """
        if image.shape[-1] in {1, 3}:  # a.k.a. if in format (H, W, C)
            image = image.permute(2, 0, 1)

        image = ((image - image.min()) / (image.max() - image.min()) - 0.5) * 2
        image = image.cuda()
        
        if resize:
            result_size = (int(image.shape[1] * self.scaling_factor), int(image.shape[2] * self.scaling_factor))
            image = transforms.Resize(result_size)(image)
        
        return image

    def __transform_calibration_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """
        Transform the calibration matrix with respect to the scaling factor.
        """
        K_prim = torch.tensor([
            [self.scaling_factor * K[0, 0], 0, self.scaling_factor * K[0, 2]],
            [0, self.scaling_factor * K[1, 1], self.scaling_factor * K[1, 2]],
            [0, 0, 1]
        ]).to(Met3rLoss.DEVICE)
        return K_prim

    def preprocess_depth(self, depth: torch.Tensor, resize: bool = True) -> torch.Tensor:
        """
        Preprocess a depth tensor for MEt3R: Make the depth in shape of (1, H, W)
        and resize with the scaling factor if needed.
        """
        depth = depth.squeeze().unsqueeze(0)
        assert depth.dim() == 3
        depth = depth.cuda()
        if resize:
            result_size = (int(depth.shape[1] * self.scaling_factor), int(depth.shape[2] * self.scaling_factor))
            depth = transforms.Resize(result_size)(depth)
        return depth

    def forward(
        self,
        gt_image: torch.Tensor,
        ood_rendered_image: torch.Tensor,
        gt_depth: torch.Tensor,
        ood_depth: torch.Tensor,
        gt_pose: torch.Tensor,
        ood_pose: torch.Tensor,
        calibration_matrix: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        use_oclusion_mask: bool = True,
        **kwargs):
        """
        Compute the MET3R loss between a ground truth image and an OOD rendered image.
        The loss is computed only on the valid pixels and areas where the images overlap.
        The score_map represents the per-pixel error between the images.

        Args:
            gt_image (torch.Tensor): Ground truth image tensor of shape (H, W, 3)
            ood_rendered_image (torch.Tensor): Out-of-distribution rendered image tensor of shape (H, W, 3)
            gt_depth (torch.Tensor): Ground truth depth tensor of shape (H, W, 1)
            ood_depth (torch.Tensor): Out-of-distribution depth tensor of shape (H, W, 1)
            gt_pose (torch.Tensor): Ground truth pose tensor of shape (4, 4)
            ood_pose (torch.Tensor): Out-of-distribution pose tensor of shape (4, 4)
            calibration_matrix (torch.Tensor): Calibration matrix tensor of shape (3, 3)
            valid_mask (torch.Tensor, optional): Binary mask indicating valid pixels of shape (H, W, 1). Defaults to None.
            use_oclusion_mask (bool, optional): Whether to use the oclusion mask. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - score (torch.Tensor): Mean MET3R score across valid pixels
                - score_map (torch.Tensor): Per-pixel MET3R score map of shape (H, W, 1)
                - overlap_mask (torch.Tensor): Binary mask indicating where images overlap, shape (H, W, 1)
        """
        self.orig_size = gt_image.shape[-3:-1]  # assumes (H, W, C)
        assert gt_image.shape == ood_rendered_image.shape
        assert gt_pose.shape == ood_pose.shape == (4, 4)
        assert calibration_matrix.shape == (3, 3)
        assert gt_image.dim() == 3
        self.__set_scaling_factor(self.orig_size)

        gt_image = self.preprocess_tensor(gt_image)
        ood_rendered_image = self.preprocess_tensor(ood_rendered_image)
        calibration_matrix = self.__transform_calibration_matrix(calibration_matrix)
        gt_depth = self.preprocess_depth(gt_depth)
        ood_depth = self.preprocess_depth(ood_depth)

        _, overlap_mask, score_map = self.metric(
            train_rgb=gt_image,
            ood_rgb=ood_rendered_image,
            train_depth=gt_depth,
            ood_depth=ood_depth,
            train_pose=gt_pose,
            ood_pose=ood_pose,
            camera_matrix=calibration_matrix,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=False,
            use_oclusion_mask=use_oclusion_mask,
            pad=self.pad,
            **kwargs)

        score_map = score_map * overlap_mask

        score_map[score_map == 1.0] = 0.0

        if max(self.orig_size) > max(score_map.shape):
            score_map = F.interpolate(score_map.unsqueeze(0), size=self.orig_size, mode='bilinear', align_corners=False).squeeze()

        if valid_mask is not None:
            score_map = score_map * valid_mask.squeeze().to(Met3rLoss.DEVICE)

        score = score_map[score_map != 0].mean()

        score_map = score_map.detach().cpu()
        return score, score_map.squeeze().unsqueeze(-1), overlap_mask.detach().cpu().squeeze().unsqueeze(-1)

    def forward_multiview_consistency(
        self,
        gt_image: torch.Tensor,
        ood_rendered_image: torch.Tensor,
        gt_depth: torch.Tensor,
        ood_depth: torch.Tensor,
        gt_pose: torch.Tensor,
        ood_pose: torch.Tensor,
        calibration_matrix: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        use_oclusion_mask: bool = True,
        **kwargs):
        """
        Experimental feature to calculate the multiview consistency RGB loss.
        """

        assert gt_image.shape == ood_rendered_image.shape
        assert gt_pose.shape == ood_pose.shape == (4, 4)
        assert calibration_matrix.shape == (3, 3)
        assert gt_image.dim() == 3

        gt_image = self.preprocess_tensor(gt_image, resize=False)
        ood_rendered_image = self.preprocess_tensor(ood_rendered_image, resize=False)
        gt_depth = self.preprocess_depth(gt_depth, resize=False)
        ood_depth = self.preprocess_depth(ood_depth, resize=False)

        l1_loss_map, overlap_mask, projections = self.metric.forward_rgb_features(
            train_rgb=gt_image,
            ood_rgb=ood_rendered_image,
            train_depth=gt_depth,
            ood_depth=ood_depth,
            train_pose=gt_pose,
            ood_pose=ood_pose,
            camera_matrix=calibration_matrix,
            return_overlap_mask=True,
            return_projections=True,
            use_oclusion_mask=use_oclusion_mask,
            **kwargs)
        
        _, gt_projections = torch.unbind(projections, dim=1)
        if valid_mask is not None:
            overlap_mask = overlap_mask * valid_mask.squeeze().to(Met3rLoss.DEVICE)
            
            
        l1_loss = l1_loss_map[overlap_mask == 1.0].sum() / (overlap_mask == 1.0).sum().clamp(min=1e-3)
        l1_loss_map = l1_loss_map.squeeze().mean(dim=-1).detach().cpu().unsqueeze(-1)
        overlap_mask = overlap_mask.detach().cpu().squeeze().unsqueeze(-1)
        gt_projections = gt_projections.squeeze().detach().cpu()
        return l1_loss, l1_loss_map, gt_projections, overlap_mask
    
    def forward_depth_mvc(self,
        gt_depth: torch.Tensor,
        ood_depth: torch.Tensor,
        gt_pose: torch.Tensor,
        ood_pose: torch.Tensor,
        calibration_matrix: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        use_oclusion_mask: bool = True,
        **kwargs):
        
        gt_depth = self.preprocess_depth(gt_depth, resize=False)
        ood_depth = self.preprocess_depth(ood_depth, resize=False)
        
        l1_loss_map, overlap_mask = self.metric.forward_depth_mvc(
            train_depth=gt_depth,
            ood_depth=ood_depth,
            camera_matrix=calibration_matrix,
            train_pose=gt_pose,
            ood_pose=ood_pose,
            return_overlap_mask=True,
            use_oclusion_mask=use_oclusion_mask,
            **kwargs)
        
        if valid_mask is not None:
            overlap_mask = overlap_mask * valid_mask.squeeze().to(Met3rLoss.DEVICE)
        
        l1_loss = l1_loss_map[overlap_mask == 1.0].sum() / (overlap_mask == 1.0).sum().clamp(min=1e-3)
        
        l1_loss_map = l1_loss_map.squeeze().detach().cpu().unsqueeze(-1)
        overlap_mask = overlap_mask.detach().cpu().squeeze().unsqueeze(-1)
        return l1_loss, l1_loss_map, overlap_mask

