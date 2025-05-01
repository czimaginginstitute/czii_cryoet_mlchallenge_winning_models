import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import zarr


# class TrainDataset(Dataset):
#     def __init__(
#             self, 
#             copick_root=None, 
#             run_names=[], 
#             classes=[], 
#             pixelsize=10.012, 
#             transforms=None, 
#             n_aug=1112, 
#             crops_per_sample=4, 
#             crop_size=(96,96,96)):
        
#         self.run_names = run_names
#         self.root = copick_root
#         self.class2id = {c: i for i, c in enumerate(classes)}
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
#         zarr_store = run.get_voxel_spacing(10).get_tomogram('denoised').zarr() # <- FSStore
#         tomogram_array = zarr.open(zarr_store, mode='r')[0]  #.transpose(2, 1, 0)
#         crop = tomogram_array[:20, :20, :20][:]  # loads chunks 256 x 256 x 256, more chuncks need more time to load    
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
#             tomogram_crop = tomogram_zarr[z0:z0+crop_z, y0:y0+crop_y, x0:x0+crop_x][:]
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

#         images = torch.stack(images)  # (crops_per_sample, C, Z, Y, X)
#         labels = torch.stack(labels)  # (crops_per_sample, C, Z, Y, X)

#         return {
#             "input": images,
#             "target": labels,
#         }


class TrainDataset(Dataset):
    def __init__(
            self, 
            copick_root=None,
            run_names: list = [],
            classes: list = [],      
            pixelsize: float = 10.012, 
            transforms=None,
            n_aug=1112
        ):
        self.run_names = run_names   # list of metadata dicts (only run_names etc.)
        self.root = copick_root
        self.class2id = {c:i for i,c in enumerate(classes)}  
        self.pixelsize = pixelsize
        self.transforms = transforms
        self.n_aug = n_aug
        self.len = len(run_names) * n_aug

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_idx = idx // self.n_aug
        run_name = self.run_names[sample_idx]

        # Lazy load tomogram
        run = self.root.get_run(run_name)
        tomogram = run.get_voxel_spacing(10).get_tomogram('denoised').numpy().transpose(2,1,0)

        locations = []
        classes = []
        for pick in run.picks:
            if pick.user_id == "curation":
                for point in pick.points:
                    locations.append([point.location.x, point.location.y, point.location.z])
                    classes.append(self.class2id[pick.pickable_object_name])

        locations = np.array(locations) / self.pixelsize
        mask = np.zeros((len(self.class2id),) + tomogram.shape[-3:])
        mask[classes, locations[:,0].astype(int), locations[:,1].astype(int), locations[:,2].astype(int)] = 1

        sample = {
            "image": tomogram,
            "label": mask,
        }

        if self.transforms:
            sample = self.transforms(sample)   # random_crop, flip, rotate, etc.

        data = {
            "input": torch.stack([item['image'] for item in sample]),
            "target": torch.stack([item['label'] for item in sample]),
        }

        return data


class CopickDataset(Dataset):
    def __init__(
            self, 
            copick_root=None,
            run_names: list = [],
            classes: list = [],      
            pixelsize: float = 10.012,
            transforms=None, 
        ):

        self.root = copick_root
        self.run_names = run_names
        self.n_classes = len(classes)
        self.class2id = {c:i for i,c in enumerate(classes)}  
        self.pixelsize = pixelsize
        self.transforms = transforms
        self.pickable_objects = {obj.name: obj.radius for obj in self.root.pickable_objects}
    
    def __len__(self):
        return len(self.run_names)
    
    def __getitem__(self, idx):
        run_name = self.run_names[idx]
        run = self.root.get_run(run_name)
        tomogram = run.get_voxel_spacing(10).get_tomogram('denoised').numpy().transpose(2,1,0)
        # pick_points = dict()
        # for pick in run.picks:
        #     if pick.user_id == "curation":
        #         points = pick.points
        #         object_name = pick.pickable_object_name
        #         radius = self.pickable_objects.get(object_name, None)
        #         if radius is None:
        #             print(f"Skipping object {object_name} as it has no radius.")
        #             continue
        #         pick_points[object_name] = {
        #             'points': np.array([[p.location.x, p.location.y, p.location.z] for p in points]) / self.pixelsize,
        #             'radius': radius
        #         }
        meta = {'run': run, 'pixelsize': self.pixelsize, 'pickable_objects': self.pickable_objects}
        data = {'meta': meta, 'input': tomogram}
        if self.transforms:
            data = self.transforms(data)       
        
        return data



# class TrainDataset(Dataset):
#     def __init__(
#             self,
#             data,
#             transforms=None,
#             n_aug=1112, 
#     ):
#         self.dataset = monai.data.Dataset(data=data, transform=train_aug)
#         self.transforms = transforms
#         self.n_aug = n_aug
#         self.len = len(data)*n_aug

#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, idx):
#         dataset = self.dataset[idx//self.n_aug]
#         data_dict = {
#             "input": torch.stack([item['image'] for item in dataset]), # [4, 1, 96, 96, 96] (minibatch, channel, z, y, x)
#             "target": torch.stack([item['label'] for item in dataset]),
#         }  

#         return data_dict




