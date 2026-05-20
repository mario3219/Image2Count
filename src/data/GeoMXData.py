from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RandomJitter, KNNGraph, Distance, LocalCartesian
import torch
import torch_geometric
import os
import squidpy as sq
import pandas as pd
import numpy as np
from anndata import AnnData
from tqdm import tqdm

class GeoMXDataset(Dataset):
    """
    Dataset of Cell Graphs and their summed expression.
    """
    def __init__(self,
                 root_dir='data/',
                 split='train',
                 raw_subset_dir='',
                 train_ratio=0.6,
                 val_ratio=0.2,
                 num_folds=1,
                 node_dropout=0.2,
                 edge_dropout=0.3,
                 pixel_pos_jitter=40,
                 n_knn=6,
                 subgraphs_per_graph=0,
                 num_hops=10,
                 label_data='label_data.csv',
                 transform=None,
                 use_embed_image=True,
                 **kwargs):
        """
        Init dataset.

        root_dir (str): Path to dir containing raw/ and processed dir
        raw_subset_dir (str): Name of dir in raw/ and processed/ containing  per ROI visual cell representations(in raw/)
        train_ratio (float): Ratio of IDs used for training
        val_ratio (float): Ratio of IDs used for validation
        num_folds (int): Number of Crossvalidation folds, val_ratio is disregarded, train_ratio is data used for Crossvalidation, 1-train_ratio is ratio of test data used over all folds
        node_dropout (float): Chance of node dropout during training
        edge_dropout (float): Chance of edge dropout during training
        pixel_pos_jitter (int): Positional jittering of nodes during training
        n_knn (int): Number of Nearest Neighbours to calculate for each cell and create edges to
        subgraphs_per_graph (int): Number of ~equally distributed subgraphs per ROI to create, use when observable SC data exists
        num_hops (int): Number of hops to create subgraphs from centoid cell
        label_data (str): .csv name in raw/ dir contaiing ROI label data
        transform (None): -
        use_embed_image (bool): Wether or not to use visual representation embedings for cells, or cell cut outs
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        assert split in ['train', 'test'], f'split must be either train or test, but is {split}'
        self.split = split
        self.raw_path = os.path.join(self.root_dir, 'raw', raw_subset_dir)
        self.processed_path = os.path.join(self.root_dir, 'processed', raw_subset_dir, self.split)
        self.label_data = label_data
        self.raw_subset_dir = raw_subset_dir

        self.num_folds = num_folds if num_folds > 1 else 1
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.pixel_pos_jitter = pixel_pos_jitter
        self.n_knn = n_knn
        self.subgraphs_per_graph = subgraphs_per_graph
        self.num_hops = num_hops
        self.use_embed_image = use_embed_image

        self.RandomJitter = RandomJitter(self.pixel_pos_jitter)
        self.KNNGraph = KNNGraph(k=self.n_knn, force_undirected=True)
        self.Distance = Distance(norm=False, cat=False)
        self.LocalCartesian = LocalCartesian()

        self.mode = self.split.upper()
        self.train = 'TRAIN'
        self.val = 'VAL'
        self.test = 'TEST'

        if not (os.path.exists(self.processed_path) and os.path.isdir(self.processed_path)):
            os.makedirs(self.processed_path)

        if os.path.exists(self.raw_path) and os.path.isdir(self.raw_path):
            self.cell_pos = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('.csv')][0]
            raw_files = pd.read_csv(os.path.join(self.root_dir, 'raw', self.label_data),
                                                      header=0,
                                                      sep=',')['ROI'].apply(lambda x: x.split('.')[0]+'_cells_embed.pt').unique().tolist()
            raw_files = [os.path.join(self.raw_path, self.split, p) for p in raw_files]
            tmp = [os.path.join(self.raw_path, split, p) for p in os.listdir(os.path.join(self.raw_path, self.split)) if p.endswith('_cells_embed.pt')]
            self.raw_files = list(set(raw_files).intersection(set(tmp)))
            self.raw_files.sort()
        
        image_name_split = pd.read_csv(self.cell_pos, header=0, sep=',', usecols=['Image'])['Image'].iloc[0].split('.')
        self.image_ending = ''
        for i in range(len(image_name_split)-1):
            self.image_ending = self.image_ending + '.' + image_name_split[i+1]

        super().__init__(self.root_dir, self.transform if transform is None else transform, None, None)

        self.data = np.array(self.processed_file_names)

        df = pd.read_csv(os.path.join(self.raw_dir, self.label_data), header=0, sep=',')
        df = df[df['ROI'].apply(lambda x: os.path.join(self.raw_path, self.split, x.split('.')[0]+'_cells_embed.pt')).isin(self.raw_files)]
        IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)  #if duplicate, take first
        un_IDs = np.unique(IDs)

        # scale factor of genes/proteins
        sf = df[df.columns[2:].values].values
        self.sf = torch.from_numpy(np.sum(sf)/np.sum(sf, axis=0)).to(torch.float32)

        total_samples = un_IDs.shape[0]
        if self.num_folds > 1 and self.mode == self.train:
            self.train_map = list(range(total_samples))
            self.val_map = list(range(total_samples))
        elif self.mode == self.train:
            train_map, val_map = torch.utils.data.random_split(torch.arange(total_samples),
                                                                        [train_ratio, val_ratio])
            self.train_map, self.val_map = np.argwhere(np.isin(IDs, un_IDs[train_map.indices])).squeeze().tolist(), np.argwhere(np.isin(IDs, un_IDs[val_map.indices])).squeeze().tolist()
            if type(self.train_map) == int:
                self.train_map = [self.train_map]
            if type(self.val_map) == int:
                self.val_map = [self.val_map]

        if self.subgraphs_per_graph > 0:
            if self.mode == self.test:
                self.train_map = list(range(total_samples))
                self.val_map = list(range(total_samples))
            map_tuple = self._create_subgraphs(self.data, self.train_map , self.val_map, IDs, un_IDs)
            self.data, self.train_map, self.val_map, IDs = map_tuple
        
        if self.num_folds > 1:
            self.current_fold = 0
            self.IDs = IDs
            self.folds = self.kFold(self.num_folds, self.IDs)

    @property
    def raw_file_names(self):
        return self.raw_files
    
    @property
    def processed_file_names(self):
        """
        return list of files should be in processed dir, if found - skip processing.
        """
        processed_filename = []
        for i, path in enumerate(self.raw_files):
            i += 1
            appendix = path.split('/')[-1].split('_cells_embed')[0]
            if len(self.raw_subset_dir) > 0:
                processed_filename.append(os.path.join(self.raw_subset_dir, self.split, f'graph_{appendix}.pt'))
            else:
                processed_filename.append(f'graph_{appendix}.pt')
        processed_filename.sort()
        return processed_filename

    def _create_subgraphs(self, data, train_map, val_map, IDs, un_IDs):
        """
        Create somewhat equally distributed square number of subgraphs of all ROIs and save.

        Parameters:
        data (np.array): Array of file names
        train_map (list): 0 or 1 depending if ROI for training
        val_map (list): 0 or 1 depending if ROI for validation
        IDs (np.array): Patient ID of sample
        un_IDs (np.array): unique Patient IDs of sample

        Return:
        data (np.array): Array of file names of subgraphs
        new_train_map (list): 0 or 1 depending if subgraph for training
        new_val_map (list): 0 or 1 depending if subgraph for validation
        new_IDs (np.array): Patient ID of samples
        """
        if not (os.path.exists(os.path.join(self.processed_path, 'subgraphs')) and os.path.isdir(os.path.join(self.processed_path, 'subgraphs'))):
            os.makedirs(os.path.join(self.processed_path, 'subgraphs'))

        new_IDs = np.ndarray(data.shape[0]*self.subgraphs_per_graph, dtype=object)
        new_data = np.ndarray(data.shape[0]*self.subgraphs_per_graph, dtype=object)
        new_train_map, new_val_map = [], []
        with tqdm(data, total=data.shape[0], desc='Creating Subgraphs...') as data:
            for g, graph_path in enumerate(data):
                graph = torch.load(os.path.join(self.processed_dir, graph_path), weights_only=False)
                xmax, xmin, ymax, ymin = torch.max(graph.pos[:,0]), torch.min(graph.pos[:,0]), torch.max(graph.pos[:,1]), torch.min(graph.pos[:,1])
                # Calculate the step sizes for x and y dimensions
                step_x = (xmax - xmin) / (self.subgraphs_per_graph ** 0.5 + 1)
                step_y = (ymax - ymin) / (self.subgraphs_per_graph ** 0.5 + 1)

                # Generate points
                points = []
                for i in range(int(self.subgraphs_per_graph ** 0.5)):
                    for j in range(int(self.subgraphs_per_graph ** 0.5)):
                        x = xmin + i * step_x + step_x / 2
                        y = ymin + j * step_y + step_y / 2
                        points.append((x, y))
                
                for p, point in enumerate(points):
                    idx = torch.argmin(torch.abs(graph.pos[:,0]-point[0]) + torch.abs(graph.pos[:,1]-point[1]))
                    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(idx.item(),
                                                                                                self.num_hops,
                                                                                                graph.edge_index,
                                                                                                relabel_nodes=True, 
                                                                                                directed=False)
                    subgraph = Data(x=graph.x[subset],
                                edge_index=edge_index,
                                edge_attr=graph.edge_attr[edge_mask],
                                pos=graph.pos[subset],
                                cellexpr=graph.cellexpr[subset],
                                y=torch.sum(graph.cellexpr[subset], axis=0))
                    torch.save(subgraph, os.path.join(self.processed_path,
                                                    'subgraphs',
                                                    f'{p:03d}'+graph_path.split('/')[-1]))
                    new_data[g*self.subgraphs_per_graph+p] = os.path.join(graph_path.split('/')[0],
                                                self.split,
                                                'subgraphs',
                                                f'{p:03d}'+graph_path.split('/')[-1])
                    new_IDs[g*self.subgraphs_per_graph+p] = IDs[g]
                    un_ID_idx = np.where(IDs[g]==un_IDs)[0][0]
                    if un_ID_idx in train_map:
                        new_train_map.append(g*self.subgraphs_per_graph+p)
                    elif un_ID_idx in val_map:
                        new_val_map.append(g*self.subgraphs_per_graph+p)
                    else:
                        raise Exception(f'Index of {graph_path} not in train/val map')
        
        data = np.array(new_data)
        new_IDs = np.array(new_IDs)
        return data, new_train_map, new_val_map, new_IDs

    def kFold(self, K, IDs):
        un_IDs = np.unique(IDs)
        total_samples = un_IDs.shape[0]
        folds = torch.utils.data.random_split(torch.arange(total_samples), [1/K]*K)
        return folds

    def set_fold_k(self):
        if self.num_folds == 1:
            pass
        elif self.current_fold == self.num_folds:
            raise Exception(f'Current fold {self.current_fold}+1 exceeds number of folds {self.num_folds}')
        else:
            un_IDs = np.unique(self.IDs)
            train_map = []
            for i, fold in enumerate(self.folds):
                if i == self.current_fold:
                    self.val_map = np.argwhere(np.isin(self.IDs, un_IDs[fold.indices])).squeeze().tolist()
                else:
                    train_map.append(np.argwhere(np.isin(self.IDs, un_IDs[fold.indices])).squeeze().tolist())
            self.train_map = np.concatenate(train_map)
            self.current_fold += 1

    def transform(self, data):
        """"
        Transform graph if training.

        Paramters:
        data (torch_geometric.data.Data): Graph

        Returns:
        torch_geometric.data.Data: Graph
        """
        if self.mode==self.train:
            y = data.y
            data.edge_index = torch.Tensor([])
            data = self.RandomJitter(data)
            data = self.KNNGraph(data)
            data = self.Distance(data)
            node_map = torch_geometric.utils.dropout_node(data.edge_index,
                                                        p=self.node_dropout,
                                                        training=self.mode==self.train)[1]
            data.edge_index, data.edge_attr = data.edge_index[:,node_map], data.edge_attr[node_map]
            edge_map = torch_geometric.utils.dropout_edge(data.edge_index,
                                                        p=self.edge_dropout,
                                                        training=self.mode==self.train)[1]
            data.edge_index, data.edge_attr = data.edge_index[:,edge_map], data.edge_attr[edge_map]
            data = torch_geometric.transforms.AddRemainingSelfLoops(attr='edge_attr', fill_value=0.0)(data)
            data.y = y
        #data = self.LocalCartesian(data)
        return data 

    def download(self):
        pass

    def process(self):
        """
        Process all cell representations per ROI into Graphs if not done already.
        """
        label = pd.read_csv(os.path.join(self.raw_dir, self.label_data), header=0, sep=',')
        df = pd.read_csv(self.cell_pos, header=0, sep=',')
        df['Centroid.X.px'] = df['Centroid.X.px'].astype(np.float32)
        df['Centroid.Y.px'] = df['Centroid.Y.px'].astype(np.float32)
        with tqdm(self.raw_paths, total=len(self.raw_paths), desc='Preprocessing Graphs') as raw_paths:
            for file in raw_paths:
                self._process_one_step(file, df, label)

    def _process_one_step(self, file, df, label):
        """
        Preprocess and create Cell Graph of ROI.

        Paramters:
        file (str): Path and file name of cell representations of ROI
        df (pandas.DataFrame): DataFrame containing Cell postions of files
        label (pandas.DataFrame): DataFrmae containing label information of ROI
        """
        file_prefix = file.split('/')[-1].split('_cells_embed')[0]
        df = df[df['Image']==file_prefix+self.image_ending]
        # Deduplicate identical cell position
        # mask = ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
        # df = df[mask]
        if df.shape[0] < 6:
            raise Exception(f'{file_prefix} has less than 6 cells!')

        counts = np.zeros((df.shape[0], 1))
        coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))
        adata = AnnData(counts, obsm={"spatial": coordinates})
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=self.n_knn)
        edge_matrix = adata.obsp["spatial_distances"]
        edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(edge_matrix)

        if self.use_embed_image:
            node_features = torch.load(file, weights_only=False)#[torch.from_numpy(mask.values)]#TODO
        else: 
            node_features = np.load(file.split('_embed')[0]+'.npy')#[mask.values] # Cant select in torch cuz uint16

        label = label[label['ROI']==file_prefix]
        label = torch.from_numpy(label.iloc[:,~label.columns.isin(['ROI', 'Patient_ID'])].sum().to_numpy()).to(torch.float32)
        cellexpr = label.clone()
        if ~df.columns.isin(['Image', 'Centroid.X.px', 'Centroid.Y.px', 'Class']).sum()>0:
            idx = ~df.columns.isin(['Image', 'Centroid.X.px', 'Centroid.Y.px', 'Class'])
            cellexpr = torch.from_numpy(df[df.columns[idx].values].values).to(torch.float32) #SC oberseved expression data
        if torch.sum(label) > 0:
            if 'Class' in df.columns:
                data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr.to(torch.float32),
                        y=label,
                        pos=torch.from_numpy(coordinates).to(torch.float32),
                        Class=df['Class'].values,
                        cellexpr=cellexpr)
            else:
                data = Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr.to(torch.float32),
                            pos=torch.from_numpy(coordinates).to(torch.float32),
                            y=label,
                            cellexpr=cellexpr)
            data = torch_geometric.transforms.AddRemainingSelfLoops(attr='edge_attr', fill_value=0.0)(data)
            data = torch_geometric.transforms.ToUndirected(merge=False)(data)
            torch.save(data, os.path.join(self.processed_path, f"graph_{file_prefix}.pt"))
        else: 
            raise Exception(f'File {file} has no Expression data in {self.label_data}!!!')

    def setMode(self, mode):
        """
        Set mode of dataset.

        Parameters:
        mode (str): mode of dataset to set to
        """
        if mode.upper() in [self.train, self.val, self.test]:
            self.mode = mode.upper()
        else:
            print(f'Mode {mode} not suported, has to be one of .train, .val .test or .embed')

    def len(self):
        """
        Set mode of dataset.
        """
        if self.mode == self.train:
            return len(self.train_map)
        elif self.mode == self.val:
            return len(self.val_map)
        elif self.mode == self.test:
            return self.data.shape[0]
        else:
            return self.data.shape[0]

    def get(self, idx):
        """
        Get Graph self.data[idx] depending on mode.

        Parameters:
        idx (int): index

        Returns:
        torch_geometric.data.Data, Cell Graph
        """
        if self.mode == self.train:
            return torch.load(os.path.join(self.processed_dir, self.data[self.train_map][idx]), weights_only=False)
        elif self.mode == self.val:
            return torch.load(os.path.join(self.processed_dir, self.data[self.val_map][idx]), weights_only=False)
        elif self.mode == self.test:
            return torch.load(os.path.join(self.processed_dir, self.data[idx]), weights_only=False)
        else:
            return torch.load(os.path.join(self.processed_dir, self.data[idx]), weights_only=False)
    
    def embed(self, model, path, device='cpu'):
        """
        Save model sc expression of all cells per ROI.

        model (torch.Module): model
        path (str): Dir to save ROI sc expression to
        device (str): device to operate on
        """
        with torch.no_grad():
            model = model.to(device)
            with tqdm(self.data.tolist(), total=self.data.shape[0], desc='Creating ROI embeddings') as data:
                for graph_path in data:
                    graph = torch.load(os.path.join(self.processed_dir, graph_path), weights_only=False)
                    cell_pred = model(graph.to(device), return_cells=True)
                    roi_pred = torch.sum(cell_pred, axis=0)
                    torch.save(roi_pred, os.path.join(path, 'roi_pred_'+graph_path.split('/')[-1]))
                    torch.save(cell_pred, os.path.join(path, 'cell_pred_'+graph_path.split('/')[-1]))
