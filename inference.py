import os, sys, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import einops
import importlib
import argparse
import zarr, time
from pathlib import Path
from copy import copy
from ctypes import *

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval

from typing import List, Tuple, Union, Any, Iterable, Optional


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import cv2
    cv2.setNumThreads(0)
except:
    print('no cv2 installed, running without')


sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")
sys.path.append("metrics")


ANGSTROMS_IN_PIXEL = 10.012

TARGET_CLASSES = (
    {
        "name": "apo-ferritin",
        "label": 0,
        "color": [0, 117, 255],
        "radius": 60,
        "map_threshold": 0.0418,
    },
    {
        "name": "beta-galactosidase",
        "label": 1,
        "color": [176, 0, 192],
        "radius": 90,
        "map_threshold": 0.0578,
    },
    {
        "name": "ribosome",
        "label": 2,
        "color": [0, 92, 49],
        "radius": 150,
        "map_threshold": 0.0374,
    },
    {
        "name": "thyroglobulin",
        "label": 3,
        "color": [43, 255, 72],
        "radius": 130,
        "map_threshold": 0.0278,
    },
    {
        "name": "virus-like-particle",
        "label": 4,
        "color": [255, 30, 53],
        "radius": 135,
        "map_threshold": 0.201,
    },
    {"name": "beta-amylase", "label": 5, "color": [153, 63, 0, 128], "radius": 65, "map_threshold": 0.035},
)

CLASS_LABEL_TO_CLASS_NAME = {c["label"]: c["name"] for c in TARGET_CLASSES}
TARGET_SIGMAS = [c["radius"] / ANGSTROMS_IN_PIXEL for c in TARGET_CLASSES]



def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a, b, c = value
        result = a, b, c

    return result


def get_volume(
    root_dir: str | Path,
    study_name: str,
    mode: str = "denoised",
    voxel_spacing_str: str = "VoxelSpacing10.000",
):
    """
    Opens a Zarr store for the specified study and mode (e.g. denoised, isonetcorrected),
    returns it as a NumPy array (fully loaded).

    :param root_dir: Base directory (e.g., /path/to/czii-cryo-et-object-identification).
    :param study_name: For example, "TS_5_4".
    :param mode: Which volume mode to load, e.g. "denoised", "isonetcorrected", "wbp", etc.
    :param voxel_spacing_str: Typically "VoxelSpacing10.000" from your structure.
    :return: A 3D NumPy array of the volume data.
    """
    # Example path:
    #   /.../ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr
    zarr_path = os.path.join(
        str(root_dir),
        "ExperimentRuns",
        study_name,
        voxel_spacing_str,
        f"{mode}.zarr",
    )

    # Open the top-level Zarr group
    store = zarr.DirectoryStore(zarr_path)
    zgroup = zarr.open(store, mode="r")

    #
    # Typically, you'll see something like zgroup[0][0][0] or zgroup['0']['0']['0']
    # for the actual volume data, but it depends on how your Zarr store is structured.
    # Let’s assume the final data is at zgroup[0][0][0].
    #
    # You may need to inspect your actual Zarr structure and adjust accordingly.
    #
    volume = zgroup[0]  # read everything into memory

    return np.asarray(volume)



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


@torch.no_grad()
def decode_detections_with_nms(
    scores: List[Tensor],
    offsets: List[Tensor],
    strides: List[int],
    min_score: Union[float, List[float]],
    class_sigmas: List[float],
    iou_threshold: float = 0.25,
    use_single_label_per_anchor: bool = True,
    use_centernet_nms: bool = False,
    pre_nms_top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detections from scores and centers with Non-Maximum Suppression (NMS)
    postprocess localization
    1. remove below a threshold
    2. deduplicate based on ranking

    :param scores: Predicted scores of shape (C, D, H, W)
    :param offsets: Predicted offsets of shape (3, D, H, W)
    :param min_score: Minimum score to consider a detection
    :param class_sigmas: Class sigmas (class radius for NMS), length = number of classes
    :param iou_threshold: Threshold above which detections are suppressed

    :return:
        - final_centers [N, 3] (x, y, z)
        - final_labels [N]
        - final_scores [N]
    """

    # Number of classes is the second dimension of `scores`
    # e.g. scores shape = (C, D, H, W)
    num_classes = scores[0].shape[0]  # the 'C' dimension

    # Allow min_score to be a single value or a list of values
    min_score = np.asarray(min_score, dtype=np.float32).reshape(-1)
    if len(min_score) == 1:
        min_score = np.full(num_classes, min_score[0], dtype=np.float32)

    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0)).squeeze(0) for s in scores] # dark filter below a threshold

    scores, centers, _ = decode_detections([s.unsqueeze(0) for s in scores], [o.unsqueeze(0) for o in offsets], strides)
    scores = scores.squeeze(0)
    centers = centers.squeeze(0)

    labels_of_max_score = scores.argmax(dim=1)

    # Prepare final outputs
    final_labels_list = []
    final_scores_list = []
    final_centers_list = []

    # NMS per class
    for class_index in range(num_classes):
        sigma_value = float(class_sigmas[class_index])  # Get the sigma for this class
        score_threshold = float(min_score[class_index])
        score_mask = scores[:, class_index] >= score_threshold  # Filter out low-scoring detections

        if use_single_label_per_anchor:
            class_mask = labels_of_max_score.eq(class_index)  # Pick out only detections of this class
            mask = class_mask & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores = scores[mask, class_index]  # shape: [Nc]
        class_centers = centers[mask]  # shape: [Nc, 3]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True, sorted=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        # Run a simple “greedy” NMS
        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)  # track suppressed

        # print(f"Predictions for class {class_index}: ", torch.count_nonzero(class_mask).item())

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            # Keep this detection
            keep_indices.append(i)

            # Suppress detections whose IoU with i is above threshold
            iou = keypoint_similarity(class_centers[i : i + 1, :], class_centers, sigma_value)

            high_iou_mask = iou > iou_threshold
            suppressed |= high_iou_mask.to(suppressed.device)

        print(f"Predictions for class {class_index} after NMS", len(keep_indices))

        # Gather kept detections for this class
        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    # Concatenate from all classes
    final_labels = torch.cat(final_labels_list, dim=0) if final_labels_list else torch.empty((0,), dtype=torch.long)
    final_scores = torch.cat(final_scores_list, dim=0) if final_scores_list else torch.empty((0,))
    final_centers = torch.cat(final_centers_list, dim=0) if final_centers_list else torch.empty((0, 3))

    print(f"Final predictions after NMS: {final_centers.size(0)}")
    return final_centers, final_labels, final_scores


def mean_std_renormalization(volume):
    """
    Renormalize the volume to have zero mean and unit variance.
    :param volume: Tensor of shape (B, C, D, H, W)
    """
    mean = volume.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = volume.std(dim=(1, 2, 3, 4), keepdim=True)
    volume = (volume - mean) / std
    return volume


def get_new_slices(slices, z_scale):
    new_slices = []
    for __s in slices:
        new_slices += [tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))]
    return new_slices



def sliding_window_inference4(
    inputs,
    predictor,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    n_classes=7,
    overlap=(0.21, 0.21, 0.21),
    z_scale=[0.5, 0.5, 0.5],
    verbose=False,
):

    image_size = inputs.shape[2:]
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims=3, overlap=overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)
    # print(len(slices))

    C = n_classes
    n_batches = np.ceil(len(slices) / sw_batch_size).astype(int)
    batch_slices = np.array_split(slices, n_batches)
    out_slices = np.array_split(get_new_slices(slices, z_scale), n_batches)

    # pred = torch.zeros((C, *image_size),dtype=res.dtype,device=res.device)
    # count = torch.zeros((C, *image_size),dtype=torch.int8,device=res.device)

    out_shape = (torch.tensor(image_size) * torch.tensor(z_scale)).long().tolist()
    out_preds = torch.zeros((C, *out_shape), dtype=inputs.dtype, device=inputs.device)
    #print(f'out_preds: {out_preds.shape}') #  torch.Size([7, 315, 315, 92])
    out_counts = torch.zeros((C, *out_shape), dtype=torch.int8, device=inputs.device)

    with torch.inference_mode():
        predictor = predictor.cuda().eval()
        for batch_idx, batch_slice in tqdm(enumerate(batch_slices), disable=1 - verbose):
            #print(f'batch_idx {batch_idx}')
            patches = torch.cat([inputs[(slice(1), slice(1)) + tuple(s)] for s in batch_slice])
            #print(patches.shape)

            p = predictor(patches.cuda())
            #print(p['logits'].shape) # penultimate logits (1, 7, 48, 48, 48) 
            for s, r in zip(out_slices[batch_idx], p['logits']):
                #print(f's {s.shape}\nr {r.shape}')  # (3, ) [7, 48, 48, 48]
                #print((slice(C),) + tuple(s))  # (slice(None, 7, None), slice(0, 48, None), slice(112, 160, None), slice(0, 48, None))
                out_preds[(slice(C),) + tuple(s)] += r
                out_counts[(slice(C),) + tuple(s)] += 1

        out_preds /= out_counts

    return out_preds  # .cpu()



def postprocess_scores_offsets_into_submission(
    scores,
    offsets,
    iou_threshold,
    output_strides,
    score_thresholds,
    study_name,
    use_centernet_nms,
    use_single_label_per_anchor,
    pre_nms_top_k: int,
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
    submission = dict(
        experiment=[],
        particle_type=[],
        score=[],
        x=[],
        y=[],
        z=[],
    )
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))
    submission = pd.DataFrame.from_dict(submission)
    return submission




######### regular torch checkpoints ############
# def get_random_torch_checkpoints(directory: str):
#     """
#     Searches for standard PyTorch checkpoint files (.pt/.pth) in the given directory and returns a random subset.

#     :param directory: Path to the directory containing PyTorch checkpoint files.
#     :param num_checkpoints: Number of models to select randomly.
#     :return: List of randomly selected checkpoint paths.
#     """
#     # Find all .pt and .pth files in the directory
#     ckpt_files = glob.glob(f"{directory}/**/*.pt", recursive=True) + glob.glob(f"{directory}/**/*.pth", recursive=True)

#     if not ckpt_files:
#         raise FileNotFoundError(f"No PyTorch checkpoint files found in {directory}")
    
#     return ckpt_files


# def load_models(cfg, checkpoint_paths, device, torch_dtype):
#     """
#     Loads multiple PyTorch models for ensemble inference.
    
#     :param checkpoint_paths: List of paths to .pt or .pth model checkpoints.
#     :param device: CUDA or CPU device for model loading.
#     :param torch_dtype: Data type for the models.
#     :return: List of loaded models.
#     """
#     models = []
    
#     for ckpt_path in checkpoint_paths:
#         checkpoint = torch.load(ckpt_path, map_location=device)
        
#         if "model" in checkpoint:
#             model_state_dict = checkpoint["model"]
#         else:
#             model_state_dict = checkpoint
        
#         Net = importlib.import_module(cfg.model).Net
#         model = Net(cfg)
#         model.load_state_dict(model_state_dict)
#         model.to(device).to(torch_dtype)
#         model.eval()  # Set to evaluation mode
#         models.append(model)

#     return models


def get_random_torch_checkpoints(directory: str):
    """
    Searches for standard PyTorch and PyTorch Lightning checkpoint files in the given directory.
    :param directory: Path to the directory containing checkpoint files.
    :return: List of checkpoint paths.
    """
    ckpt_files = (
        glob.glob(f"{directory}/**/*.pt", recursive=True)
        + glob.glob(f"{directory}/**/*.pth", recursive=True)
        + glob.glob(f"{directory}/**/*.ckpt", recursive=True)
    )

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {directory}")

    return ckpt_files


def load_models(cfg, checkpoint_paths, device, torch_dtype):
    """
    Loads multiple PyTorch or Lightning model checkpoints for ensemble inference.
    
    :param checkpoint_paths: List of checkpoint file paths (.pt/.pth/.ckpt).
    :param device: Device to load the models onto.
    :param torch_dtype: Model dtype (e.g., torch.float32 or torch.float16).
    :return: List of loaded model instances.
    """
    models = []

    for ckpt_path in checkpoint_paths:
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Extract state_dict from Lightning checkpoints
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # plain PyTorch

        # Remove prefixes like "model." or "net."
        clean_state_dict = {
            k.replace("model.", "").replace("net.", ""): v
            for k, v in state_dict.items()
        }

        # Dynamically import model
        Net = importlib.import_module(cfg.model).Net
        model = Net(cfg)

        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device).to(dtype=torch_dtype)
        model.eval()
        models.append(model)

    return models



def main_inference_entry_point_regular(
    cfg,
    model_dir: str,  # Directory containing PyTorch .pt/.pth model checkpoints
    score_thresholds,
    device_id: int,
    world_size: int = 2,
    tiles_per_dim=(1, 9, 9),
    output_strides: list = (2,),
    window_size=(192, 128, 128),
    iou_threshold: float = 0.85,
    use_weighted_average: bool = True,
    use_centernet_nms: bool = True,
    use_single_label_per_anchor: bool = False,
    use_z_flip_tta: bool = False,
    use_y_flip_tta: bool = False,
    use_x_flip_tta: bool = False,
    pre_nms_top_k: int = 16536,
    batch_size=1,
    num_workers=0,
    torch_dtype=torch.float16,
    data_path="./data",
    tomogram_list=[],
    mode = "denoised"
):
    device_id = int(device_id)
    torch_device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")  

    # Select random PyTorch checkpoints for ensemble
    ckpt_fns = get_random_torch_checkpoints(model_dir)

    # Load selected PyTorch models
    models = load_models(cfg, ckpt_fns, device=torch_device, torch_dtype=torch_dtype)

    path = Path(data_path)
    studies_path = path / "ExperimentRuns"
    studies = tomogram_list if tomogram_list else list(sorted(os.listdir(studies_path)))
    # Not using distributed inference
    # print(f"studies\n{studies}")
    # studies = studies[device_id::world_size]
    print("Process got", len(studies), "to process")

    submissions = []

    for study_name in studies:
        study_volume = get_volume(
            root_dir=path,
            study_name=study_name,
            mode=mode
        )

        study_sub = predict_volume_regular(
            models=models,
            volume=study_volume,
            study_name=study_name,
            output_strides=output_strides,
            window_size=window_size,
            tiles_per_dim=tiles_per_dim,
            use_weighted_average=use_weighted_average,
            use_centernet_nms=use_centernet_nms,
            use_single_label_per_anchor=use_single_label_per_anchor,
            pre_nms_top_k=pre_nms_top_k,
            use_z_flip_tta=use_z_flip_tta,
            use_y_flip_tta=use_y_flip_tta,
            use_x_flip_tta=use_x_flip_tta,
            score_thresholds=score_thresholds,
            iou_threshold=iou_threshold,
            batch_size=batch_size,
            num_workers=num_workers,
            device_id=device_id,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        submissions.append(study_sub)
        print(f'submissions\n{submissions}')

    submission = pd.concat(submissions)
    return submission


@torch.no_grad()
def predict_volume_regular(
    *,
    volume: np.ndarray,
    models: list,
    output_strides: list,
    window_size: tuple,
    tiles_per_dim: tuple,
    study_name: str,
    score_thresholds: Union[float, list],
    iou_threshold,
    batch_size,
    num_workers,
    use_weighted_average,
    use_centernet_nms,
    use_single_label_per_anchor,
    device_id: int,
    torch_device: str,
    torch_dtype,
    pre_nms_top_k,
    use_z_flip_tta: bool,
    use_y_flip_tta: bool,
    use_x_flip_tta: bool,
):
    """
    Slide window infrerence with TTA.
    """
    # Convert volume to (B, C, D, H, W)
    volume_tensor = (
        mean_std_renormalization(torch.from_numpy(volume).permute(2, 1, 0)[None, None])
        .to(torch_dtype)
        .to(torch_device)
    )

    img_flipped = torch.flip(volume_tensor, [2, 3, 4])
    ensemble_prediction = None
    particle_ids = [0, 2, 3, 4, 5, 1]
    count = 0

    for i, model in enumerate(models):
        model = model.eval().to(torch_device)
        print(model)
        pred = sliding_window_inference4(
            inputs=volume_tensor,
            predictor=model,
            roi_size=(96, 96, 96),
            sw_batch_size=batch_size,
            n_classes=7,
            overlap=(0.21, 0.21, 0.21),
            z_scale=[0.5, 0.5, 0.5],
            verbose=False
        )
        pred += torch.flip(
            sliding_window_inference4(
                inputs=img_flipped,
                predictor=model,
                roi_size=(96, 96, 96),
                sw_batch_size=batch_size,
                n_classes=7,
                overlap=(0.21, 0.21, 0.21),
                z_scale=[0.5, 0.5, 0.5],
                verbose=False
            ), [1, 2, 3]
        )

        if ensemble_prediction is None:
            ensemble_prediction = pred
        else:
            ensemble_prediction += pred
        count += 2

    ensemble_prediction /= count
    ensemble_prediction = F.interpolate(
        ensemble_prediction.softmax(0)[particle_ids][None],
        (630, 630, 184),
        mode="trilinear"
    )[0]
    ensemble_prediction = F.interpolate(
        ensemble_prediction[None],
        (630 // 2, 630 // 2, 184 // 2),
        mode="trilinear"
    )[0]
    ensemble_prediction = ensemble_prediction.permute(0, 3, 2, 1)

    fake_offsets = torch.zeros_like(ensemble_prediction[0:3])

    submission = postprocess_scores_offsets_into_submission(
        scores=[ensemble_prediction],
        offsets=[fake_offsets],
        iou_threshold=iou_threshold,
        output_strides=output_strides,
        score_thresholds=score_thresholds,
        study_name=study_name,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    return submission



def main_regular(cfg, 
                 device_id=0, 
                 checkpoints='./checkpoints',
                 data_path="./data",
                 output_dir = './output',
                 tomogram_list = [],
                 mode = "denoised"

    ):
    submission = main_inference_entry_point_regular(
        cfg,
        checkpoints,  # Directory containing PyTorch .pt/.pth model checkpoints
        score_thresholds=[
            0.16,
            0.25,
            0.13,
            0.19,
            0.18,
            0.5,
        ],  
        # [0.16, 0.25, 0.12, 0.19, 0.18, 0.5] LB 794
        # [0.255,0.235,0.16 ,0.255,0.225, 0.5], # LB: 784 V4 OOF Computed CV score: 0.8295528641195601 std: 0.01879723638715648
        device_id=device_id,
        data_path=data_path,
        tomogram_list = tomogram_list,
        mode = mode
    )


    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    submission.to_csv(f"{output_dir}/submission_shard_{device_id}.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("-c", "--config", help="config filename")
    parser.add_argument("-d", "--device", type=int, help="config filename")
    parser.add_argument("-i", "--data_folder", type=str, default="./data", help="data folder for inference. Example:  data_folder/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr")
    parser.add_argument("-m", "--mode", type=str, default="denoised", help="reconstruction mode")
    parser.add_argument("-p", "--checkpoints", type=str, default="./checkpoints", help="checkpoints folder (multiple checkpoints will create a model soup)")
    parser.add_argument("-D", "--debug", action='store_true', help="debugging True/ False")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="outputs")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    sys.path.append("src/czii_cryoet_models/configs")
    sys.path.append("src/czii_cryoet_models/data")
    sys.path.append("src/czii_cryoet_models/metrics")
    sys.path.append("src/czii_cryoet_models/models")
    sys.path.append("src/czii_cryoet_models/postprocess")
    sys.path.append("src/czii_cryoet_models/utils")
    print(importlib.import_module(parser_args.config))
    cfg = copy(importlib.import_module(parser_args.config).cfg)  

    public = "TS_100_3,TS_101_5,TS_102_2,TS_103_4,TS_103_5,TS_105_5,TS_105_7,TS_106_7,TS_107_1,TS_109_2,TS_109_3,TS_109_5,TS_111_2,TS_112_6,TS_114_4,TS_115_4,TS_115_7,TS_11_7,TS_12_3,TS_12_9,TS_13_9,TS_14_1,TS_14_2,TS_14_4,TS_14_6,TS_15_1,TS_1_2,TS_1_3,TS_1_4,TS_1_5,TS_1_7,TS_1_8,TS_25_6,TS_27_1,TS_27_7,TS_2_3,TS_30_2,TS_31_8,TS_33_4,TS_37_1,TS_37_8,TS_43_9,TS_44_4,TS_45_1,TS_46_8,TS_47_2,TS_47_7,TS_48_5,TS_4_3,TS_52_3,TS_53_7,TS_58_6,TS_58_7,TS_58_9,TS_59_1,TS_59_8,TS_5_2,TS_5_8,TS_60_6,TS_60_7,TS_60_8,TS_63_6,TS_63_7,TS_65_4,TS_65_8,TS_65_9,TS_67_2,TS_67_3,TS_67_6,TS_67_7,TS_68_4,TS_68_7,TS_6_7,TS_70_8,TS_70_9,TS_71_3,TS_71_7,TS_71_8,TS_74_1,TS_74_3,TS_74_8,TS_76_3,TS_76_4,TS_76_8,TS_79_4,TS_80_5,TS_80_8,TS_81_1,TS_81_4,TS_83_1,TS_84_5,TS_84_7,TS_85_1,TS_86_4,TS_86_8,TS_86_9,TS_87_4,TS_87_5,TS_87_6,TS_87_7,TS_88_3,TS_89_2,TS_89_5,TS_90_2,TS_90_3,TS_90_4,TS_91_7,TS_91_9,TS_93_5,TS_93_9,TS_94_3,TS_94_5,TS_94_6,TS_94_7,TS_96_4,TS_96_9,TS_97_7,TS_98_1,TS_98_4,TS_9_2,TS_9_3"
    public = public.split(",")
    private = "TS_100_4,TS_100_6,TS_100_7,TS_100_9,TS_101_1,TS_101_4,TS_101_6,TS_101_7,TS_101_9,TS_102_1,TS_102_4,TS_102_7,TS_102_8,TS_103_1,TS_103_2,TS_103_3,TS_103_9,TS_104_1,TS_104_2,TS_104_3,TS_104_4,TS_104_6,TS_104_9,TS_105_2,TS_106_1,TS_106_4,TS_106_5,TS_107_3,TS_107_5,TS_108_1,TS_108_3,TS_108_4,TS_108_6,TS_108_7,TS_108_8,TS_108_9,TS_109_1,TS_109_4,TS_109_6,TS_109_8,TS_109_9,TS_110_2,TS_110_3,TS_110_8,TS_111_1,TS_111_5,TS_111_8,TS_111_9,TS_112_4,TS_112_5,TS_113_4,TS_114_1,TS_114_2,TS_114_3,TS_114_5,TS_115_1,TS_115_3,TS_115_5,TS_115_6,TS_115_8,TS_116_4,TS_116_5,TS_116_6,TS_116_9,TS_117_7,TS_118_7,TS_11_1,TS_11_3,TS_11_9,TS_120_2,TS_12_1,TS_12_2,TS_12_5,TS_13_5,TS_13_8,TS_14_5,TS_14_8,TS_15_2,TS_15_3,TS_15_6,TS_15_7,TS_15_8,TS_15_9,TS_1_1,TS_1_6,TS_1_9,TS_24_3,TS_24_4,TS_24_5,TS_24_6,TS_24_8,TS_25_1,TS_25_2,TS_25_3,TS_25_4,TS_25_5,TS_25_7,TS_25_8,TS_26_1,TS_26_4,TS_26_6,TS_26_7,TS_27_3,TS_27_4,TS_27_5,TS_27_9,TS_28_3,TS_28_5,TS_29_7,TS_29_9,TS_2_2,TS_2_4,TS_2_5,TS_2_6,TS_2_9,TS_30_7,TS_30_9,TS_31_1,TS_31_3,TS_31_6,TS_32_3,TS_32_5,TS_33_7,TS_34_2,TS_34_5,TS_34_7,TS_34_8,TS_35_3,TS_35_6,TS_35_7,TS_35_9,TS_37_2,TS_37_4,TS_37_6,TS_39_1,TS_39_5,TS_39_7,TS_3_3,TS_3_4,TS_3_5,TS_42_7,TS_43_2,TS_43_3,TS_43_5,TS_43_7,TS_44_1,TS_44_2,TS_44_3,TS_44_5,TS_44_7,TS_44_8,TS_44_9,TS_45_2,TS_45_4,TS_45_5,TS_45_6,TS_46_1,TS_46_5,TS_46_9,TS_47_3,TS_47_4,TS_47_6,TS_47_8,TS_48_1,TS_48_2,TS_48_4,TS_48_9,TS_49_2,TS_49_5,TS_49_6,TS_4_4,TS_4_5,TS_4_6,TS_4_7,TS_4_9,TS_52_4,TS_52_5,TS_56_7,TS_58_1,TS_58_8,TS_59_2,TS_59_3,TS_59_4,TS_59_5,TS_59_6,TS_59_7,TS_5_1,TS_5_3,TS_5_5,TS_5_9,TS_60_1,TS_60_2,TS_60_3,TS_60_4,TS_60_5,TS_61_1,TS_61_2,TS_61_4,TS_61_5,TS_61_6,TS_61_7,TS_61_8,TS_62_1,TS_62_2,TS_62_4,TS_62_5,TS_62_7,TS_62_8,TS_63_2,TS_63_3,TS_63_5,TS_63_8,TS_63_9,TS_64_3,TS_64_9,TS_65_2,TS_65_6,TS_67_4,TS_67_8,TS_68_5,TS_68_8,TS_68_9,TS_69_1,TS_69_4,TS_69_5,TS_69_7,TS_69_8,TS_6_1,TS_6_8,TS_6_9,TS_70_1,TS_70_2,TS_70_3,TS_70_4,TS_70_5,TS_70_7,TS_71_2,TS_71_4,TS_71_5,TS_71_9,TS_72_2,TS_72_3,TS_72_7,TS_72_8,TS_73_7,TS_73_8,TS_74_5,TS_75_1,TS_75_2,TS_75_7,TS_76_5,TS_76_6,TS_76_9,TS_77_1,TS_77_2,TS_77_3,TS_77_5,TS_77_6,TS_78_1,TS_78_2,TS_78_4,TS_78_5,TS_78_8,TS_78_9,TS_79_5,TS_79_6,TS_79_9,TS_7_1,TS_7_2,TS_7_4,TS_7_8,TS_80_1,TS_80_6,TS_81_2,TS_81_6,TS_81_8,TS_82_1,TS_82_2,TS_82_3,TS_82_4,TS_82_5,TS_83_3,TS_83_5,TS_83_6,TS_83_8,TS_84_1,TS_84_8,TS_85_2,TS_85_6,TS_85_8,TS_86_1,TS_86_2,TS_86_5,TS_86_6,TS_87_1,TS_87_2,TS_87_3,TS_87_8,TS_87_9,TS_88_1,TS_88_7,TS_88_8,TS_88_9,TS_89_1,TS_89_3,TS_89_4,TS_89_6,TS_89_8,TS_8_7,TS_90_1,TS_90_5,TS_90_6,TS_90_7,TS_91_2,TS_91_3,TS_91_5,TS_91_8,TS_92_1,TS_92_2,TS_92_4,TS_92_8,TS_93_1,TS_93_2,TS_93_3,TS_93_4,TS_93_6,TS_93_7,TS_93_8,TS_94_1,TS_94_4,TS_94_8,TS_95_1,TS_95_3,TS_95_5,TS_95_6,TS_95_8,TS_95_9,TS_96_2,TS_96_3,TS_96_5,TS_96_7,TS_96_8,TS_97_2,TS_97_3,TS_97_4,TS_97_5,TS_97_6,TS_98_2,TS_98_3,TS_98_5,TS_98_6,TS_98_7,TS_98_9,TS_99_1,TS_99_2,TS_99_3,TS_99_4,TS_99_6,TS_99_7,TS_9_4,TS_9_5,TS_9_7,TS_9_8,TS_9_9"
    private = private.split(",")

    start = time.time()
    main_regular(cfg, 
                 checkpoints = parser_args.checkpoints,
                 device_id = parser_args.device,
                 data_path = parser_args.data_folder,
                 output_dir = parser_args.output_dir,
                 mode = parser_args.mode,
                 #tomogram_list = private
                 )
    print(f'The inference process takes {time.time()-start} s to finish.')
