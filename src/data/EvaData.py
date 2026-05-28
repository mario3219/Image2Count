import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
from src.utils.eva_kit.constant import marker_to_gene

class EvaDataset():
    """
    Dataset for single-image patchified data.
    """
    def __init__(self,
                 conf,
                 root_dir='data/raw',
                 raw_subset_dir='',
                 split='train',
                 **args):
        
        assert split in ['train', 'test'], f'split must be either train or test, but is {split}'
 
        self.work_dir = os.path.join(os.getcwd(), root_dir, 'raw', raw_subset_dir)
        self.img_dir = os.path.join(self.work_dir,split)
        self.cells_path = [os.path.join(self.img_dir, p) for p in os.listdir(self.img_dir) if p.lower().endswith('_cells.npy')]
        self.conf = conf

        channel_names = args['channel_names']
        channel_mask = [True]*len(channel_names)
        for idx, channel_name in enumerate(channel_names):
            if channel_name not in marker_to_gene.keys():
                print(f"WARNING! {channel_name} is not in GenePT embeddings and will be masked!")
                channel_mask[idx] = False
        self.channel_names = np.array(channel_names)[channel_mask]
               
    def save_embed_data(self, model, device='cpu', batch_size=256):
        """
        Save model representations of all cells per ROI.

        model (torch.Module): KRONOS model
        device (str): device to operate on
        batch_size (int): Number of cells to extract representations from at once
        """
        
        model.eval()
        with torch.no_grad():
            for path in tqdm(self.cells_path, desc='Save embeddings'):
                if not os.path.exists(os.path.join(path,path.split('.')[0]+'_embed.pt')):
                    sample = torch.from_numpy(np.load(path)) # (B, C, H, W)
                    embed = torch.empty((sample.shape[0],self.conf.pm.dim), dtype=torch.float32)
                    bms = [self.channel_names.copy() for _ in range(sample.shape[0])]

                    num_batches = (sample.shape[0]+batch_size-1) // batch_size
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            idx_start = batch_idx*batch_size
                            idx_end = batch_idx*batch_size+batch_size
                        else:
                            idx_start = batch_idx*batch_size
                            idx_end = len(embed)

                        image_out, _ = model.model.forward_encoder(sample[idx_start:idx_end].to(device), bms[idx_start:idx_end])
                        image_cls = image_out[:,:,0,:]
                        image_cls = image_cls.squeeze(1)
                        batch_size = image_cls.size(0)
                        feat = image_cls.view(batch_size, -1)
                        embed[idx_start:idx_end] = feat

                    torch.save(embed, os.path.join(path, path.split('.')[0]+'_embed.pt'))
                    del sample
                    del embed
