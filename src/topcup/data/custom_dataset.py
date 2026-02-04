import numpy as np
import zarr
from pathlib import Path
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
            self, 
            root_path,
            pattern: int="*.zarr",     
            pixelsize: float = 10.012,
            pickable_objects: dict={},
            transforms=None, 
        ):
        root = Path(root_path)
        self.tomograms = list(root.rglob(pattern))
        self.pixelsize = pixelsize
        self.transforms = transforms
        self.pickable_objects = pickable_objects
    
    def __len__(self):
        return len( self.tomograms)
    
    def __getitem__(self, idx):
        tomogram_path = self.tomograms[idx]
        try:
            with zarr.open(tomogram_path) as zf:
                tomogram = np.array(zf[0]).transpose(2,1,0) 
        except Exception as e:
            print(e)

        run_name = str(tomogram_path).split('/')[-1].split('.')[0]
        meta = {'run_name': run_name, 'pixelsize': self.pixelsize, 'pickable_objects': self.pickable_objects, 'dim': tomogram.shape}
        data = {'meta': meta, 'input': tomogram, 'dataset_type': 'custom', 'has_ground_truth': False}
        if self.transforms:
            data = self.transforms(data)       
        
        return data
