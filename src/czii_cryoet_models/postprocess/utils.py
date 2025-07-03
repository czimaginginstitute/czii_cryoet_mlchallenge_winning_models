import numpy as np
import einops
import torch
from torch import Tensor
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval
from collections import defaultdict
import torch.nn.functional as F
import copy
import pandas as pd
from czii_cryoet_models.postprocess.metric_old import score, process_run
from czii_cryoet_models.postprocess.constants import ANGSTROMS_IN_PIXEL, CLASS_INDEX_TO_CLASS_NAME, TARGET_SIGMAS, WEIGHTS

from typing import List, Tuple, Union, Any, Iterable, Optional


def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a, b, c = value
        result = a, b, c

    return result



def anchors_for_offsets_feature_map(offsets, stride):
    z, y, x = torch.meshgrid(
        torch.arange(offsets.size(-3), device=offsets.device),
        torch.arange(offsets.size(-2), device=offsets.device),
        torch.arange(offsets.size(-1), device=offsets.device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0)
    anchors = anchors.float().add_(0.5).mul_(stride)

    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors


def keypoint_similarity(pts1, pts2, sigmas):
    """
    Compute similarity between two sets of keypoints
    :param pts1: ...x3
    :param pts2: ...x3
    """
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)  # []
    e: Tensor = d / (2 * sigmas**2)
    iou = torch.exp(-e)
    return iou


def centernet_heatmap_nms(
    scores,
    kernel: Union[int, Tuple[int, int, int]] = 3,
    # kernel: Union[int, Tuple[int, int, int]] = (5,3,3)
):
    kernel = as_tuple_of_3(kernel)
    pad = (kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2

    maxpool = torch.nn.functional.max_pool3d(scores, kernel_size=kernel, padding=pad, stride=1)

    mask = scores == maxpool
    peaks = scores * mask
    return peaks


def decode_detections(logits: Tensor | List[Tensor], offsets: Tensor | List[Tensor], strides: int | List[int]):
    """
    Decode detections from logits and offsets
    :param logits: Predicted logits B C D H W
    :param offsets: Predicted offsets B 3 D H W
    :param anchors: Stride of the network

    :return: Tuple of probas and centers:
             probas - B N C
             centers - B N 3

    """
    if torch.is_tensor(logits):
        logits = [logits]
    if torch.is_tensor(offsets):
        offsets = [offsets]
    if isinstance(strides, int):
        strides = [strides]

    anchors = [anchors_for_offsets_feature_map(offset, s) for offset, s in zip(offsets, strides)]

    logits_flat = []
    centers_flat = []
    anchors_flat = []

    for logit, offset, anchor in zip(logits, offsets, anchors):
        centers = anchor + offset

        logits_flat.append(einops.rearrange(logit, "B C D H W -> B (D H W) C"))
        centers_flat.append(einops.rearrange(centers, "B C D H W -> B (D H W) C"))
        anchors_flat.append(einops.rearrange(anchor, "B C D H W -> B (D H W) C"))

    logits_flat = torch.cat(logits_flat, dim=1)
    centers_flat = torch.cat(centers_flat, dim=1)
    anchors_flat = torch.cat(anchors_flat, dim=1)

    return logits_flat, centers_flat, anchors_flat



def mean_std_renormalization(volume):
    mean = volume.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = volume.std(dim=(1, 2, 3, 4), keepdim=True)
    volume = (volume - mean) / std
    return volume

def get_new_slices(slices, z_scale):
    new_slices = []
    for __s in slices:
        new_slices += [tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))]
    return new_slices

def sliding_window(
        inputs, 
        predictor, 
        roi_size=(96, 96, 96), 
        sw_batch_size=1, 
        n_classes=7, 
        overlap=(0.21, 0.21, 0.21), 
        z_scale=[0.5, 0.5, 0.5], 
        verbose=False,
):
    image_size = inputs.shape[-3:]
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims=3, overlap=overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    C = n_classes
    n_batches = np.ceil(len(slices) / sw_batch_size).astype(int)
    batch_slices = np.array_split(slices, n_batches)
    out_slices = np.array_split(get_new_slices(slices, z_scale), n_batches)

    out_shape = (torch.tensor(image_size) * torch.tensor(z_scale)).long().tolist()
    out_preds = torch.zeros((C, *out_shape), dtype=inputs.dtype, device=inputs.device)
    out_counts = torch.zeros((C, *out_shape), dtype=torch.int8, device=inputs.device)
    out_loss = 0
    out_loss_count = 0

    with torch.no_grad():
        predictor = predictor.cuda().eval()
        for batch_idx, batch_slice in enumerate(batch_slices):
            patches = torch.cat([inputs[(slice(1), slice(1)) + tuple(s)] for s in batch_slice])
            p = predictor(patches.cuda())
            if "loss" in p:
                loss = p['loss']
                out_loss += loss.item()
                out_loss_count += 1
            for s, r in zip(out_slices[batch_idx], p['logits']):
                out_preds[(slice(C),) + tuple(s)] += r
                out_counts[(slice(C),) + tuple(s)] += 1
            
        out_preds /= out_counts
        if out_loss_count > 0:
            out_loss /= out_loss_count

    return out_preds, out_loss


def decode_detections_with_nms(
        scores, 
        offsets, 
        strides, 
        min_score, 
        class_sigmas, 
        iou_threshold=0.25, 
        use_single_label_per_anchor=True, 
        use_centernet_nms=True, 
        pre_nms_top_k=None
        ):
    num_classes = scores[0].shape[0]
    min_score = np.asarray(min_score, dtype=np.float32).reshape(-1)
    if len(min_score) == 1:
        min_score = np.full(num_classes, min_score[0], dtype=np.float32)

    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0)).squeeze(0) for s in scores]

    scores, centers, _ = decode_detections([s.unsqueeze(0) for s in scores], [o.unsqueeze(0) for o in offsets], strides)
    scores = scores.squeeze(0)
    centers = centers.squeeze(0)
    labels_of_max_score = scores.argmax(dim=1)

    final_labels_list, final_scores_list, final_centers_list = [], [], []
    for class_index in range(num_classes):
        sigma_value = float(class_sigmas[class_index])
        score_threshold = float(min_score[class_index])
        score_mask = scores[:, class_index] >= score_threshold

        if use_single_label_per_anchor:
            class_mask = labels_of_max_score.eq(class_index)
            mask = class_mask & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores = scores[mask, class_index]
        class_centers = centers[mask]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True, sorted=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            keep_indices.append(i)
            iou = keypoint_similarity(class_centers[i:i+1], class_centers, sigma_value)
            suppressed |= iou > iou_threshold

        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    final_labels = torch.cat(final_labels_list, dim=0) if final_labels_list else torch.empty((0,), dtype=torch.long)
    final_scores = torch.cat(final_scores_list, dim=0) if final_scores_list else torch.empty((0,))
    final_centers = torch.cat(final_centers_list, dim=0) if final_centers_list else torch.empty((0, 3))

    return final_centers, final_labels, final_scores



def postprocess_scores_offsets_into_submission(
    scores,
    offsets,
    iou_threshold,
    output_strides,
    score_thresholds,
    run_name,
    use_centernet_nms,
    use_single_label_per_anchor,
    pre_nms_top_k,
    submission_pick_points,
    mode = "validation",
):
    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    topk_scores = topk_scores.float().cpu().numpy()
    top_coords = topk_coords_px.float().cpu().numpy() * ANGSTROMS_IN_PIXEL
    topk_clses = topk_clses.cpu().numpy()
    print(f'top_coords:\n{top_coords}topk_clses:\n{topk_clses}topk_scores:\n{topk_scores}')

    submission = dict(
        experiment=[],
        particle_type=[],
        score=[],
        x=[],
        y=[],
        z=[],
    )
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        #points[CLASS_INDEX_TO_CLASS_NAME[int(cls)]].append([float(coord[0], float(coord[1], float(coord[2]))])
        submission["experiment"].append(run_name)
        submission["particle_type"].append(CLASS_INDEX_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))
        submission_pick_points[CLASS_INDEX_TO_CLASS_NAME[int(cls)]]['points'].append([float(coord[0]), float(coord[1]), float(coord[2])])
    
    for object_name, item in submission_pick_points.items():
        submission_pick_points[object_name]['points'] = np.array(item['points'])

    if mode == "inference":
        df = pd.DataFrame.from_dict(submission)
        df.to_csv("submission.csv", index=False)

    return submission, submission_pick_points


def postprocess_pipeline(pred, metas, D, H, W, mode="validation"):
    # No TTA during validation step (to keep it fast)
    # Apply softmax and interpolate back to original size (7, 315, 315, 92) -> (1, 7, 630, 630, 184)
    pred = pred.softmax(0)[:-1][None]  # (1, 7, 315, 315, 92) 'trilinear' interpolation in x,y,z directions is only available for 5D tensors
    pred = F.interpolate(pred, (D, H, W), mode="trilinear")[0] # (7, 630, 630, 184)
    # Downsample again if needed (based on your inference code)
    pred = F.interpolate(pred[None], (D // 2, H // 2, W // 2), mode="trilinear")[0] # (7, 315, 315, 92)
    pred = pred.permute(0, 3, 2, 1)  # (C, W, H, D)

    # Create fake offsets (zeros) for CenterNet decoding
    fake_offsets = torch.zeros_like(pred[0:3])

    all_results = {}
    print(f'metas: {len(metas)}')
    for meta in metas:
        run = meta['run']
        run_name = run.name
        pixelsize = meta['pixelsize']
        pickable_objects = meta['pickable_objects']
        gt_pick_points = dict()
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
                    gt_pick_points[object_name]['points'].append([p.location.x/pixelsize, p.location.y/pixelsize, p.location.z/pixelsize])
        
        for object_name, item in gt_pick_points.items():
            gt_pick_points[object_name]['points'] = np.array(item['points'])
        
        submission, submission_pick_points = postprocess_scores_offsets_into_submission(
            scores=[pred],
            offsets=[fake_offsets],
            iou_threshold=0.85,
            output_strides=(2,),
            score_thresholds=[0.16, 0.25, 0.13, 0.19, 0.18, 0.5],
            run_name=run_name,
            use_centernet_nms=True,
            use_single_label_per_anchor=False,
            pre_nms_top_k=16536,
            submission_pick_points=submission_pick_points,
            mode = mode
        )

        all_results[run_name] = process_run(gt_pick_points, submission_pick_points, pickable_objects)
    print(all_results)
    
    return score(all_results, WEIGHTS)