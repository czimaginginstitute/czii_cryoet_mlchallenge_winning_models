import einops
import numpy as np
import pandas as pd

import torch
from torch import Tensor
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval

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


def get_final_submission(
        submission: pd.DataFrame, 
        score_thresholds: dict={}, 
        output_dir: str=''
    ) -> pd.DataFrame:
    submission_pp = []
    #print(f'submission\n{submission}')
    for p, th in score_thresholds.items():
        submission_pp += [submission[(submission['particle_type']==p) & (submission['conf']>th)].copy()]
    
    submission_pp = pd.concat(submission_pp)
    submission_pp = submission_pp.sort_values(by='experiment')
    submission_pp = submission_pp.drop_duplicates(subset=['experiment', 'x', 'y', 'z'])  # by default, keep first
    if str(output_dir):
        print(f'Save predicted results in {str(output_dir)}/val_pred_df_seed.csv')
        submission_pp.to_csv(f"{str(output_dir)}/val_pred_df_seed.csv",index=False)
    
    return submission_pp


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

