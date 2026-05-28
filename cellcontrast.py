import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for image model")

    # Arguments for Image preprocessing
    parser.add_argument("--preprocess_dir", type=str, default="data/raw/p2106",
                        help="Directory in which .tiff files are for preprocessing")
    parser.add_argument("--preprocess_channels", type=str, default="",
                        help="Indices of channels to preprocess, seperated by , and empty if all channels")
    parser.add_argument("--calc_mean_std", action="store_true", default=False,
                        help="Wether or not to calculate mean and std of cell cut outs")
    parser.add_argument("--cell_cutout", type=int, default=20,
                        help="Size*Size cutout of cell, centered on Centroid Cell position")
    parser.add_argument("--preprocess_workers", type=int, default=1,
                        help="""Number of threads to use for loading data. Increasing `num_workers` past 24 may result
                        in large increases in CPU memory footprint. Only recommended for systems with
                        ``>64 GB`` RAM.""")
    parser.add_argument("--image_preprocess", action="store_true", default=False,
                        help="Wether or not to preprocess images via ZScore normalisation")

    # General Model Arguments
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Wether or not to run NNs deterministicly")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random computations")
    parser.add_argument("--root_dir", type=str, default="data/",
                        help="Where to find the raw/ and processed/ dirs")
    parser.add_argument("--raw_subset_dir", type=str, default="TMA1_preprocessed",
                        help="How the subdir in raw/ and processed/ is called")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of elements per Batch")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for which to train")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of worker processes to be used(loading data etc)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate of model")
    parser.add_argument("--weight_decay", type=float, default=5e-6,
                        help="Weight decay of optimizer")
    parser.add_argument("--early_stopping", type=int, default=100,
                        help="Number of epochs after which to stop model run without improvement to val loss")
    parser.add_argument("--output_name", type=str, default="out/models/image_contrast.pt",
                        help="Path/name of moel for saving")

    # Jonathan edit
    parser.add_argument("--foundation_model",type=str, default="",
                        help="Which foundation model to use")
    parser.add_argument("--channel_names",nargs='+',type=str, default="",
                        help="""
                        For '--foundation_model deepcell,':
                        Channel names, example '--channel_names CD3 CD8 CD20'
                        """)
    parser.add_argument("--token_size",type=int, default=14,
                        help="""
                        For '--foundation_model kronos,':
                        token_size**2 total amount of tokens to accommodate.
                        token_size has to be in range 1-14.
                        """)
    parser.add_argument("--mpp",type=float, default=0.399,
                        help="""
                        For '--foundation_model deepcell,':
                        Microns per pixel, passed as float.
                        """)
    parser.add_argument("--cell_ids",type=str, default="",
                        help="""
                        For --foundation_model deepcell/kronos/eva,
                        Used instead of {MEASUREMENTS}.csv
                        """)

    # Arguments for image model
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Number of Epochs in which learning rate gets increased")
    parser.add_argument("--embed", type=int, default=256,
                        help="Linear net size used to embed data")
    parser.add_argument("--contrast", type=int, default=124,
                        help="Linear net size on which to calculate the contrast loss")
    parser.add_argument("--crop_factor", type=float, default=0.5,
                        help="Cell Image crop factor for Image augmentation")
    parser.add_argument("--resnet", type=str, default="18",
                        help="What ResNet model to choose, on of 18, 34, 50 and 101")
    parser.add_argument("--n_clusters_image", type=int, default=1,
                        help="Number of Clusters to use for KMeans when only use when >= 1")
    parser.add_argument("--train_image_model", action="store_true", default=False,
                        help="Wether or not to train the Image model")
    parser.add_argument("--embed_image_data", action="store_true", default=False,
                        help="Wether or not to embed data with a given Image model")
    return parser.parse_args()


def main(**args):
    if args['image_preprocess']:
        if args['foundation_model'] == 'deepcell':
            from src.utils.deepcell_kit.image_funcs import image_preprocess as ImagePreprocessDCT
            ImagePreprocessDCT(path=args['preprocess_dir'], 
                            channel_names=args['channel_names'],
                            cell_cutout=args['cell_cutout'],
                            mpp=args['mpp'],
                            batch_size=args['batch_size'],
                            ids_path=args['cell_ids'])
        elif args['foundation_model'] == 'kronos':
            from src.utils.kronos_kit.image_funcs import image_preprocess as ImagePreprocessKRONOS
            ImagePreprocessKRONOS(path=args['preprocess_dir'], 
                            channel_names=args['channel_names'],
                            cell_cutout=args['cell_cutout'],
                            batch_size=args['batch_size'],
                            ids_path=args['cell_ids'])
        elif args['foundation_model'] == 'eva':
            from src.utils.eva_kit.image_funcs import image_preprocess as ImagePreprocessEVA
            ImagePreprocessEVA(path=args['preprocess_dir'], 
                            channel_names=args['channel_names'],
                            cell_cutout=args['cell_cutout'],
                            batch_size=args['batch_size'],
                            ids_path=args['cell_ids'])
        else:
            from src.utils.image_preprocess import image_preprocess as ImagePreprocess
            ImagePreprocess(path=args['preprocess_dir'], 
                            img_channels=args['preprocess_channels'],
                            do_mean_std=args['calc_mean_std'],
                            cell_cutout=args['cell_cutout'],
                            num_processes=args['preprocess_workers'])
    if args['train_image_model']:
        from src.run.CellContrastTrain import train as ImageTrain
        ImageTrain(**args)
    if args['embed_image_data']:
        from src.run.CellContrastEmbed import embed as CellContrastEmbed
        CellContrastEmbed(**args)

if __name__ == '__main__':
    args = vars(parse_args())
    main(**args)
