import pandas as pd
import torch
import copy
import numpy as np
from collections import defaultdict



def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points; Returns the maximum point scores within the radius, 
        with the suppressing baclground is zero. 
    """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool3d(x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)



def predict_volume_val(
        run_name,
        logits,
        submission_pick_points, 
        classes=["apo-ferritin", "beta-amylase", "beta-galactosidase", "ribosome", "thyroglobulin", "virus-like-particle"],
        radius={"apo-ferritin":60, "beta-amylase":65, "beta-galactosidase":90, "ribosome":150, "thyroglobulin":130, "virus-like-particle":135},
        pixelsize=10.012,
    ):
    
    img = logits[None].cuda()  # (1, 7, 315, 315, 92)
    img = torch.nn.functional.interpolate(img, size=(630, 630, 184), mode='trilinear', align_corners=False) # upsize to (1, 7, 630, 630, 184)
    w, h, d = img.shape[-3:] 
    resized_img = torch.nn.functional.interpolate(img, size=(w//2,h//2,d//2), mode='trilinear', align_corners=False) # downsize to (1, 7, 315, 315, 92)
    preds = resized_img[0].softmax(0)[:-1] # (6, 315, 315, 92), remove the background channel
    pred_df = []
    for i,p in enumerate(classes):
        print(f'adding {p}')
        p1 = preds[i][None].cuda()
        y = simple_nms(p1, nms_radius=int(0.5 * radius[p]/pixelsize))  # score of maximum points
        kps = torch.where(y > 0)
        coords = torch.stack(kps[1:], -1) * pixelsize * 2 # (x,y,z) coordinates in the origianl tomogram size
        conf = y[kps]  # probability of the predicted points
        pred_df_ = pd.DataFrame(coords.cpu().numpy(),columns=['x','y','z'])
        pred_df_['particle_type'] = p
        pred_df_['experiment'] = run_name
        pred_df_['conf'] = conf.cpu().numpy()
        pred_df += [pred_df_]
    pred_df = pd.concat(pred_df)
    pred_df = pred_df[(pred_df['x']<6300) & (pred_df['y']<6300)& (pred_df['z']<1840) & (pred_df['conf']>0.01)].copy()  # hardcore thresholding, keep prob > 0.01

    for i, row in pred_df.iterrows():
        coord = [row['x'], row['y'], row['z']]
        label = row['particle_type']
        submission_pick_points[label]['points'].append([float(coord[0]), float(coord[1]), float(coord[2])])

    return pred_df, submission_pick_points



def postprocess_pipeline_val(pred, metas):
    # No TTA during validation step (to keep it fast)
    # Apply softmax and interpolate back to original size (7, 315, 315, 92) -> (1, 7, 630, 630, 184)
    print(f'metas: {len(metas)}')
    gt_dfs = []
    submission_dfs = []
    for meta in metas:
        run = meta['run']
        run_name = run.name
        pickable_objects = meta['pickable_objects']
        gt_pick_points = dict()
        gt_df = defaultdict(list)
        for object_name, radius in pickable_objects.items():
            gt_pick_points[object_name] = {
                'points': [],
                'radius': radius
            }
        submission_pick_points = copy.deepcopy(gt_pick_points)
        for pick in run.picks:
            if pick.user_id == "curation":
                points = pick.points
                for p in points:
                    object_name = pick.pickable_object_name
                    gt_pick_points[object_name]['points'].append([p.location.x, p.location.y, p.location.z])
                    gt_df['x'].append(p.location.x)
                    gt_df['y'].append(p.location.y)
                    gt_df['z'].append(p.location.z)
                    gt_df['particle_type'].append(object_name) 
                    gt_df['experiment'].append(run_name)      
        
        gt_df = pd.DataFrame(gt_df)
        gt_dfs.append(gt_df)
        for object_name, item in gt_pick_points.items():
            gt_pick_points[object_name]['points'] = np.array(item['points'])
        
        print(f'now predict TS {run_name}')
        submission_df, submission_pick_points1 = predict_volume_val(
            run_name,
            pred,
            submission_pick_points
        )
        submission_dfs.append(submission_df)

    final_gt_df = pd.concat(gt_dfs, ignore_index=True)
    final_submission_df = pd.concat(submission_dfs, ignore_index=True)
    
    return final_gt_df, final_submission_df
    
