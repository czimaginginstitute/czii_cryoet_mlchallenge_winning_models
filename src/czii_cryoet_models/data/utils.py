import torch
import numpy as np


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train_collate_fn(batch):
    batch = [item for item in batch if item]
    if not batch:
        return None 

    keys = batch[0].keys()
    batch_dict = {key:torch.cat([b[key] for b in batch]) for key in keys}
    
    return batch_dict


def collate_fn(batch):
    batch = [item for item in batch if item]
    if not batch:
        return None
    
    collated = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values  # leave as list
    return collated


def get_copick_tomogram(run, pixelsize, recon_type):
    voxel_spacing_obj = None
    try:
        voxel_spacing_obj = run.get_voxel_spacing(str(pixelsize))
        if voxel_spacing_obj is None:
            print(f"Warning: get_voxel_spacing returned None for run {run.run_name} with pixelsize {pixelsize}.")
            return None
    except Exception as e:
        print(f"Error getting voxel spacing for run {run.run_name} (pixelsize: {pixelsize}): {e}. Returning None.")
        return None

    tomograms_list = None
    try:
        tomograms_list = voxel_spacing_obj.get_tomograms(str(recon_type))
        if not tomograms_list: # Checks for None or empty list
            print(f"Warning: get_tomograms returned an empty list or None for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}).")
            return None
    except AttributeError as e:
        print(f"Error calling get_tomograms on voxel_spacing_obj for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}): {e}.")
        return None
    except Exception as e:
        print(f"Error getting tomograms list for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}): {e}.")
        return None

    tomogram = None
    try:
        tomogram = tomograms_list[0].numpy().transpose(2,1,0)
        
    except IndexError:
        # This specifically catches if tomograms_list was not empty but tomograms_list[0] failed
        print(f"Warning: tomograms_list was unexpectedly empty after initial check for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}).")
        return None
    except AttributeError as e:
        # This catches if tomograms_list[0] doesn't have .numpy() or .transpose()
        print(f"Error with numpy/transpose on tomogram object for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}): {e}.")
        return None
    except Exception as e:
        # General catch-all for any other unexpected errors in this final step
        print(f"An unexpected error occurred during final tomogram processing for run {run.run_name} (pixelsize: {pixelsize}, recon_type: {recon_type}): {e}. Returning None.")
        return None
    
    return tomogram