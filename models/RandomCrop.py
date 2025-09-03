import torch
import numpy as np 
from datasets.data_transforms import PointcloudRotate

def crop(pts, crop_rate_min):
    if crop_rate_min == 1.0:    # for rebuttal
        crop_rate = 1.0
    else:
        crop_rate = np.random.uniform(crop_rate_min, 1.0)

    
    batch_size, N, _ = pts.shape 
    centers = torch.zeros((batch_size, 3)).to(pts.device)

    ret_pts = []
    for i in range(batch_size):
        center_point = pts[i, np.random.choice(pts.shape[1]), :]
        distances = torch.norm(pts[i, :, :] - center_point, dim=1) 
        _, indices = torch.topk(-distances, int(pts.shape[1] * crop_rate))
        selected_pts = pts[i, indices, :].clone() 
        selected_pts_min = selected_pts.min(dim=0).values   # (3, )
        selected_pts_max = selected_pts.max(dim=0).values   # 
        
        if crop_rate_min == 1.0:
            # no normalize only translate when do not use crop 
            selected_pts_scaled = selected_pts - (selected_pts_min + selected_pts_max) / 2
        else:
            selected_pts_std = (selected_pts - selected_pts_min) / (selected_pts_max - selected_pts_min + 1e-6)
            selected_pts_scaled = selected_pts_std * 2 - 1      
        centers[i] = (selected_pts_min + selected_pts_max) / 2                          
        ret_pts.append(selected_pts_scaled) 

    ret_pts = torch.stack(ret_pts) 
    return ret_pts, centers
        
rotate_obj = PointcloudRotate()
def pointRandomCrop(pts, rotate=True, crop_rate_min=None):  
    """
    Input the point clouds and output two decoupled view/point clouds (Decoupled views generation)

    input: pts
        (bs, N, 3)
    
    output: 
        pts1, center1
        pts2, center2
        bs, M, 3
        bs, m, 3
    """
    assert crop_rate_min is not None
    pts1, center1 = crop(pts, crop_rate_min)
    pts2, center2 = crop(pts, crop_rate_min)
    angle1, angle2 = 0, 0
    
    if rotate:
        angle1, pts1 = rotate_obj(pts1, angles_ret_needed=True)     
        angle2, pts2 = rotate_obj(pts2, angles_ret_needed=True)
    
    return center2 - center1, pts1, pts2, angle1, angle2