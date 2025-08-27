import numpy as np
import torch
from torch.utils.data import Dataset
from czii_cryoet_models.data.utils import get_copick_tomogram
from collections import defaultdict
import zarr


# class TrainDataset(Dataset):
#     def __init__(
#             self, 
#             copick_root=None, 
#             run_names=[], 
#             pixelsize=10.012, 
#             transforms=None, 
#             n_aug=1112, 
#             crops_per_sample=4, 
#             crop_size=(96,96,96)):
        
#         self.run_names = run_names
#         self.root = copick_root
#         self.class2id = {p.name:i for i,p in enumerate(self.root.pickable_objects)} 
#         self.pixelsize = pixelsize
#         self.transforms = transforms
#         self.n_aug = n_aug
#         self.crops_per_sample = crops_per_sample  # <--- number of random crops per tomogram
#         self.crop_size = crop_size
#         self.len = len(run_names) * n_aug

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         sample_idx = idx // self.n_aug
#         run_name = self.run_names[sample_idx]

#         run = self.root.get_run(run_name)     
#         zarr_store = run.get_voxel_spacing(10).get_tomograms('denoised')[0].zarr() # <- FSStore
#         tomogram_array = zarr.open(zarr_store, mode='r')[0]  #.transpose(2, 1, 0)
#         #crop = tomogram_array[:20, :20, :20][:]  # loads chunks 256 x 256 x 256, more chuncks need more time to load    
#         shape =  tomogram_array.shape
#         crop_z, crop_y, crop_x = self.crop_size

#         images = []
#         labels = []

#         for _ in range(self.crops_per_sample):
#             # Random crop starting point
#             z0 = np.random.randint(0, shape[0] - crop_z)
#             y0 = np.random.randint(0, shape[1] - crop_y)
#             x0 = np.random.randint(0, shape[2] - crop_x)

#             # Load only small crop
#             tomogram_crop = tomogram_array[z0:z0+crop_z, y0:y0+crop_y, x0:x0+crop_x][:]
#             #tomogram_crop = tomogram_crop.transpose(2, 1, 0)  

#             # Create small local mask
#             mask_crop = np.zeros((len(self.class2id),) + (crop_z, crop_y, crop_x))

#             for pick in run.picks:
#                 if pick.user_id == "curation":
#                     class_idx = self.class2id[pick.pickable_object_name]
#                     for point in pick.points:
#                         x, y, z = point.location.x / self.pixelsize, point.location.y / self.pixelsize, point.location.z / self.pixelsize
#                         x, y, z = int(x), int(y), int(z)
#                         if (z0 <= z < z0+crop_z) and (y0 <= y < y0+crop_y) and (x0 <= x < x0+crop_x):
#                             mask_crop[class_idx, z-z0, y-y0, x-x0] = 1

#             sample = {
#                 "image": tomogram_crop,
#                 "label": mask_crop,
#             }

#             if self.transforms:
#                 sample = self.transforms(sample)

#             images.append(sample["image"])
#             labels.append(sample["label"])

#         data = {
#             "input": torch.stack([item['image'] for item in sample]),
#             "target": torch.stack([item['label'] for item in sample]),
#         }

#         return data


class TrainDataset(Dataset):
    def __init__(
            self, 
            copick_root=None,
            run_names: list=[],
            pixelsize: float=10.012,
            recon_type: str='denoised',
            user_id: str='curation', 
            transforms=None,
            n_aug=1112,
            crop_radius=5, # in pixels; TODO if None, use points only; if < 1.0, create masks using a ratio of the radius; if > 1.0, create masks using the radius.
        ):
        self.run_names = [run_name for run_name in run_names if run_name]   
        self.root = copick_root
        # skip non-particles and particles do not have a radius
        self.pickable_objects ={}
        for obj in self.root.pickable_objects:
            if obj.is_particle and obj.radius:
                self.pickable_objects[obj.name] = obj.radius

        self.class2id = {p:i for i,p in enumerate(self.pickable_objects.keys())}
        self.pixelsize = pixelsize
        self.recon_type = recon_type
        self.user_id = user_id
        self.transforms = transforms
        self.n_aug = n_aug
        self.len = len(self.run_names) * n_aug
        self.crop_radius = crop_radius  # in pixels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_idx = idx // self.n_aug
        run_name = self.run_names[sample_idx]
        try:
            run = self.root.get_run(run_name)
            if run is None:
                print(f"{run_name} does not exist. Returning None.")
                return None 
        except Exception as e:
            print(f"Error getting run for run_name {run_name}. Returning None.")
            return None

        tomogram = get_copick_tomogram(run, self.pixelsize, self.recon_type)
        if tomogram is None:
            return None
        
        locations = []
        classes = []
        for pick in run.picks:
            if pick.user_id == self.user_id and pick.pickable_object_name in self.class2id.keys():
                for point in pick.points:
                    locations.append([point.location.x, point.location.y, point.location.z])
                    classes.append(self.class2id[pick.pickable_object_name])

        locations = np.array(locations) / self.pixelsize
        if self.crop_radius is not None:
            locations, classes = self.generate_nonoverlapping_locations(locations, self.crop_radius, classes, tomogram.shape)

        mask = np.zeros((len(self.class2id),) + tomogram.shape[-3:])
        mask[classes, locations[:,0].astype(int), locations[:,1].astype(int), locations[:,2].astype(int)] = 1

        sample = {
            "input": tomogram,
            "target": mask,
        }

        if self.transforms:
            sample = self.transforms(sample)   # random_crop, flip, rotate, etc.

        data = {
            "input": torch.stack([item["input"] for item in sample]),
            "target": torch.stack([item["target"] for item in sample]),
        }

        return data

    @staticmethod
    def generate_nonoverlapping_locations(locations, radius, classes, bounding_box):
        """
        Args:
            locations: (N, 3) array of integer [x, y, z] center positions
            radius: float, radius in voxel units
            classes: list or array of length N, containing class label for each location
            bounding_box: tuple (W, H, D)

        Returns:
            coords: (M, 3) array of [x, y, z] voxel coordinates
            class_list: list of length M, corresponding class for each voxel
        """
        W, H, D = bounding_box
        mask_accum = np.zeros((D, H, W), dtype=np.uint16)

        r = int(np.ceil(radius))
        offset_range = np.arange(-r, r + 1)
        dz, dy, dx = np.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
        offsets = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        distances = np.linalg.norm(offsets, axis=1)
        sphere_offsets = offsets[distances <= radius]

        voxel_to_class = {}  # use dict to map voxel → class

        for center, cls in zip(locations, classes):
            points = center + sphere_offsets

            valid = (
                (points[:, 0] >= 0) & (points[:, 0] < W) &
                (points[:, 1] >= 0) & (points[:, 1] < H) &
                (points[:, 2] >= 0) & (points[:, 2] < D)
            )
            points = points[valid].astype(int)

            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            np.add.at(mask_accum, (z, y, x), 1)

            for px, py, pz in zip(x, y, z):
                key = (px, py, pz)
                if key not in voxel_to_class:
                    voxel_to_class[key] = cls  # first one wins

        # Final filtering: keep only voxels with count == 1
        z, y, x = np.where(mask_accum == 1)
        coords = np.stack([x, y, z], axis=1)

        # Collect classes for each valid coord
        class_list = [voxel_to_class[(x_, y_, z_)] for x_, y_, z_ in coords]

        return coords, class_list


class CopickDataset(Dataset):
    def __init__(
            self, 
            copick_root=None,
            run_names: list=[],     
            pixelsize: float=10.012,
            recon_type: str='denoised',
            user_id: str='curation',
            transforms=None,
            has_ground_truth=True, 
        ):

        self.root = copick_root
        self.run_names = [run_name for run_name in run_names if run_name]  
        self.pixelsize = pixelsize
        self.recon_type = recon_type
        self.user_id = user_id
        self.transforms = transforms
        # skip non-particles and particles do not have a radius
        self.pickable_objects ={}
        for obj in self.root.pickable_objects:
            if obj.is_particle and obj.radius:
                self.pickable_objects[obj.name] = obj.radius

        self.has_ground_truth = has_ground_truth
    
    def __len__(self):
        return len(self.run_names)
    
    def __getitem__(self, idx):
        run_name = self.run_names[idx]

        # Lazy load tomogram
        try:
            run = self.root.get_run(run_name)
            if run is None:
                print(f"{run_name} does not exist. Returning None.")
                return None 
        except Exception as e:
            print(f"Error getting run for run_name {run_name}. Returning None.")
            return None

        tomogram = get_copick_tomogram(run, self.pixelsize, self.recon_type)
        if tomogram is None:
            return None

        meta = {'run': run, 
                'pixelsize': self.pixelsize, 
                'pickable_objects': self.pickable_objects, 
                'dim': tomogram.shape,
                'user_id': self.user_id
        }
        data = {'meta': meta, 'input': tomogram, 'dataset_type': 'copick', 'has_ground_truth': self.has_ground_truth}
        if self.transforms:
            data = self.transforms(data)       
        
        return data






