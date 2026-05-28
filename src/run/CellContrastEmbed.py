import torch
from src.data.CellContrastData import EmbedDataset
from src.models.CellContrastModel import ContrastiveLearning
from src.utils.utils import load
from src.utils.utils import set_seed
import os

def embed(**args):
    """
    Embed visual representations of cells.

    Parameters:
    image_dir (str): Path to dir in which torch.tensors of cell cut outs are
    model_name (str): Path and name of model torch save dict
    args (dict): Arguments
    """
    batch_size = args['batch_size']
    seed = args['seed']
    foundation_model = args['foundation_model']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    if foundation_model == 'deepcell':
 
        from src.data.DeepCellData import PatchDataset as DeepCellDataset
        train_dataset = DeepCellDataset(split='train',
                                     save_embed_data=True,
                                     **args)
        test_dataset = DeepCellDataset(split='test',
                                     save_embed_data=True,
                                     **args)

        from src.models import DeepCell
        model = DeepCell(
                n_filters=256,  # Default arguments from DEEPCELL
                n_heads=4,      
                n_domains=9,
                embed_size=256,
                **args
                ).to(device, torch.float32)
        model.load_state_dict(torch.load(args['output_name'], map_location=device))

    elif foundation_model == 'kronos':

        from src.data.KRONOSData import KRONOSDataset
        train_dataset = KRONOSDataset(split='train',
                                     save_embed_data=True,
                                     **args)
        test_dataset = KRONOSDataset(split='test',
                                     save_embed_data=True,
                                     **args)

        from src.models import KRONOS
        model = KRONOS(     # Default arguments from KRONOS,
            img_size=224,
            patch_size=16,
            embed_dim=384,
            stride_size=16,
            num_markers=512,
            init_values=1.0e-05,
            ffn_layer='mlp',
            block_chunks=4,
            num_register_tokens=16,
            ).to(device, torch.float32)
        model.load_state_dict(torch.load(args['output_name'], map_location=device))

    elif foundation_model == 'eva':

        from omegaconf import OmegaConf     # Default arguments from EVA
        conf = OmegaConf.load(
                os.path.join(os.getcwd(),"src","utils","eva_kit","config.yaml")
                )

        from src.data.EvaData import EvaDataset
        train_dataset = EvaDataset(conf=conf,
                                   split='train',
                                   save_embed_data=True,
                                   **args)
        test_dataset = EvaDataset(conf=conf,
                                  split='test',
                                  save_embed_data=True,
                                  **args)
        from src.models.EvaModel import EvaMAE
        model = EvaMAE(conf)
        model.load_state_dict(torch.load(args['output_name'], map_location=device)['state_dict'])
 
    else:

        train_dataset = EmbedDataset(split='train',
                                     save_embed_data=True,
                                     **args)
        test_dataset = EmbedDataset(split='test',
                                    save_embed_data=True,
                                    **args)

        model = ContrastiveLearning(channels=train_dataset.img_shape[0],
                                    **args).to(device, torch.float32)
        model.load_state_dict(load(args['output_name'], save_keys='model', device=device))
        model.mode = 'embed'
    
    model.eval()
    train_dataset.save_embed_data(model, device=device, batch_size=batch_size)
    test_dataset.save_embed_data(model, device=device, batch_size=batch_size)
