import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd
from src.utils.utils import per_gene_corr, total_corr, per_gene_mi
import os

def get_true_graph_expression_dict(path):
    """
    Create a dict and populate it with name of graphs and ROI expression.

    Parameters:
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression
    """
    path = os.path.join(os.getcwd(), path)
    graph_paths = [p for p in os.listdir(path) if 'graph' in p and p.endswith('pt')]
    value_dict = {}
    for graph_p in graph_paths:
        graph = torch.load(os.path.join(path, graph_p), map_location='cpu', weights_only=False)
        value_dict[graph_p] = {'y': graph.y.numpy()}
        if 'Class' in graph.to_dict().keys():
            value_dict[graph_p]['cell_class'] = graph.Class
    return value_dict

def get_predicted_graph_expression(value_dict, path):
    """
    Add predicted ROI expression to value_dict.

    Parameters:
    value_dict: dict of file name keys and corresponding ROI expression
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression(true and predicted)
    """
    path = os.path.join(os.getcwd(), path)
    roi_pred_paths = [p for p in os.listdir(path) if p.startswith('roi_pred')]
    for roi_pred_p in roi_pred_paths:
        value_dict[roi_pred_p.split('roi_pred_')[1]]['roi_pred'] = torch.load(os.path.join(path, roi_pred_p), 
                                                                            map_location='cpu',
                                                                            weights_only=False).squeeze().detach().numpy()
    return value_dict

def get_predicted_cell_expression(value_dict, path):
    """
    Add predicted sc ROI expression to value_dict, return cell shape metrics.

    Parameters:
    value_dict: dict of file name keys and corresponding ROI expression
    path (str): Path to dir containing ROI graphs

    Return:
    dict: dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    tuple: Number of cells, genes/proteins per cell
    """
    path = os.path.join(os.getcwd(), path)
    cell_pred_paths = [p for p in os.listdir(path) if p.startswith('cell_pred')]
    num_cells = 0
    for roi_pred_p in cell_pred_paths:
        value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'] = torch.load(os.path.join(path, roi_pred_p),
                                                                                map_location='cpu',
                                                                                weights_only=False).squeeze().detach().numpy()
        num_cells += value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[0]
    cell_shapes = (num_cells, value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[-1])
    return value_dict, cell_shapes

def get_patient_ids(label_data, keys):
    """
    Calculate numpy array of IDs corresponding to sorted file names of ROIs(value_dict keys), and gene/protein names.

    Parameters:
    label_data (str): Name of .csv in data/raw/ containing label information of ROIs(IDs specificly)
    keys (list): List of value_dict keys, aka ROI graph file names

    Return:
    np.array: IDs of sorted value_dict keys
    np.array: Gene/Protein names
    """
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'raw', label_data), header=0, sep=',')
    IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)
    exps = df.columns.values[2:]

    if len(keys) != IDs.shape[0]:
        tmp = np.ndarray((len(keys)), dtype=object)
        for i_key in range(len(keys)):
            tmp[i_key] = str(df[df['ROI']==keys[i_key].split('graph_')[-1].split('.')[0]]['Patient_ID'].values[0])
        IDs = tmp
    return IDs, exps

def get_bulk_expression_of(value_dict, IDs, exps, key='y'):
    """
    Create scanpy.AnnData obj of ROI expressions(NOT SC!)

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    key (str): key of ROI expression to select

    Return:
    sc.AnnData: obj of ROI expression with corresponding file names, IDs,
    """
    rois = list(value_dict.keys())
    rois.sort()
    rois_np = np.array(rois)
    adata = sc.AnnData(np.zeros((len(rois), value_dict[rois[0]][key].shape[0])))
    adata.obs['ID'] = str(-1)
    adata.var_names = exps
    files = np.array([])

    i = 0
    for id in np.unique(IDs).tolist():  #TODO: when subgraphs what then? how to automate instead of manual label creation?
        id_map = IDs==id
        id_keys = rois_np[id_map].tolist()
        for id_key in id_keys:
            adata.X[i] = value_dict[id_key][key]
            adata.obs.loc[str(i), 'ID'] = str(id)
            files = np.concatenate((files, np.array([id_key])))
            i += 1
    adata.obs['files'] = files
    return adata

def visualize_bulk_expression(value_dict, IDs, exps, name, key='y'):
    """
    Analysis of ROIs

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    key (str): key of ROI expression to select
    """
    adata = get_bulk_expression_of(value_dict, IDs, exps, key)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=20 if 20 < adata.var_names.shape[0] else adata.var_names.shape[0])
    sc.pl.highly_variable_genes(adata, save=name+'.png', show=False)
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    sc.pl.umap(adata, color=['ID', 'leiden'], save=name+'.png', show=False)
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', show=False)
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=name+'.png', show=False)

def visualize_cell_expression(value_dict, IDs, exps, name, figure_dir, cell_shapes, spearman_genes, select_cells=50000):
    """
    Create scanpy.AnnData obj of sc expressions, perform analysis and save plots.

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    name (str): name of .h5ad file to save scanpy.AnnData
    figure_dir (str): Path to save figures to
    cell_shapes (tuple): Tuple of ints(number of cells, number og gene/proteins)
    select_cells (int): Number of cells to analyse, if 0 select all, otherwise random specified subset
    """
    if os.path.exists('out/'+name+'.h5ad'):
        adata = sc.read_h5ad('out/'+name+'.h5ad')
    else:
        rois = list(value_dict.keys())
        rois.sort() # VERY IMPORTANT!!! IDs correspond to sorted value_dict keys
        rois_np = np.array(rois)
        counts = np.empty(cell_shapes, dtype=np.float32)
        cell_class = None
        ids = np.array([])
        files = np.array([])

        i = 0
        num_cells = 0
        key = 'cell_pred'
        # Build up counts array, ids, files such that every cell has correct corresponding ids and files asociated with cell ROI
        for id in np.unique(IDs).tolist():
            id_map = IDs==id
            id_keys = rois_np[id_map].tolist()  # Selects all ROI names corresponding to id
            for id_key in id_keys:
                tmp_counts = value_dict[id_key][key]
                counts[num_cells:num_cells+tmp_counts.shape[0],:] = tmp_counts
                if num_cells != 0:
                    if ('cell_class' in value_dict[id_key].keys()) and cell_class is not None:
                        cell_class = np.concatenate((cell_class, value_dict[id_key]['cell_class']))
                else:
                    if ('cell_class' in value_dict[id_key].keys()):
                        cell_class = value_dict[id_key]['cell_class']
                ids = np.concatenate((ids, np.array([id]*value_dict[id_key][key].shape[0])))
                files = np.concatenate((files, np.array([id_key]*value_dict[id_key][key].shape[0])))
                num_cells += tmp_counts.shape[0]
                i += 1
        counts = np.array(counts)

        cell_index = np.arange(counts.shape[0])
        if select_cells and counts.shape[0] > select_cells:
            cell_index = np.random.default_rng(42).choice(np.arange(counts.shape[0]), size=select_cells, replace=False)
        
        adata = sc.AnnData(counts)
        if cell_class is not None:
            cell_class = np.array(cell_class)
            adata.obs['cell_class'] = cell_class
        adata.obs['ID'] = ids
        adata.obs['files'] = files
        adata.obs['leiden'] = -1
        adata.var_names = exps
        adata.var['spearman_genes'] = adata.var_names.isin(spearman_genes)
        if not os.path.exists('out/'+name+'_all.h5ad'):
            adata.write('out/'+name+'_all.h5ad')
        
        if spearman_genes.shape[0]<2:
            print(f'Num. of significant spearman genes is {spearman_genes.shape[0]}<2, not performing analysis: Do manually!')
            return

        adata = sc.AnnData(counts[cell_index])
        if cell_class is not None:
            cell_class = np.array(cell_class)
            adata.obs['cell_class'] = cell_class[cell_index]
        adata.obs['ID'] = ids[cell_index]
        adata.obs['files'] = files[cell_index]
        adata.var_names = exps
        adata.var['spearman_genes'] = adata.var_names.isin(spearman_genes)
        
        adata.layers['counts'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['logs'] = adata.X.copy()
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,
                                    n_top_genes=20 if 20 < adata.var_names.shape[0] else adata.var_names.shape[0])
        
        sc.pp.scale(adata)
        sc.pp.pca(adata,
                  svd_solver='arpack', 
                  n_comps=np.sum(adata.var['spearman_genes'].values)-1 if np.sum(adata.var['spearman_genes'].values)-1<100 else 100,
                  chunked=True,
                  chunk_size=50000,
                  mask_var=adata.var['spearman_genes'].values)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2)

        sc.tl.rank_genes_groups(adata,
                                'leiden',
                                method='wilcoxon',
                                show=False,
                                layer='logs',
                                mask_var=adata.var['spearman_genes'])

        adata.write('out/'+name+'.h5ad')
    
    #with plt.rc_context():
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(figure_dir, f'highly_varible_genes_{name}.png'))
    plt.close()

    sc.pl.umap(adata, color='ID', show=False, legend_loc=None)
    plt.savefig(os.path.join(figure_dir, f'umap_{name}_ID.png'))
    plt.close()

    sc.pl.umap(adata, color='leiden', show=False)
    plt.savefig(os.path.join(figure_dir, f'umap_{name}_cluster.png'))
    plt.close()
    sc.pl.umap(adata, color='leiden',
               show=False, add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False)
    plt.savefig(os.path.join(figure_dir, f'umap_{name}_cluster_named.png'))
    plt.close()

    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
    plt.savefig(os.path.join(figure_dir, f'rank_genes_group_{name}.png'))
    plt.close()
    sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True, show=False, layer='logs', n_genes=5)
    plt.savefig(os.path.join(figure_dir, f'rank_genes_group_{name}_heatmap.png'))
    plt.close()

    sc.pl.heatmap(adata, adata.var_names, groupby='leiden', show=False, layer='logs')
    plt.savefig(os.path.join(figure_dir, f'heatmap_{name}.png'))
    plt.close()
    sc.pl.violin(adata, adata.var['highly_variable'].index[adata.var['highly_variable'].values].values, groupby='leiden', show=False, layer='logs')
    plt.savefig(os.path.join(figure_dir, f'violin_highly_varible_{name}.png'))
    plt.close()

    if 'cell_class' in adata.obs.columns.values.tolist() and not adata.obs['cell_class'].isna().values.sum():
        confusion_matrix = np.zeros((len(np.unique(adata.obs['leiden'])), len(np.unique(adata.obs['cell_class']))))

        # Fill the matrix based on the relationships between categories
        for i, category_a in enumerate(np.unique(adata.obs['leiden'])):
            for j, category_b in enumerate(np.unique(adata.obs['cell_class'])):
                count = np.sum((adata.obs['leiden'] == category_a) & (adata.obs['cell_class'] == category_b))
                confusion_matrix[i, j] = count

        # Create a heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=np.unique(adata.obs['cell_class']), yticklabels=np.unique(adata.obs['leiden']))
        plt.xlabel('Categories')
        plt.ylabel('Leiden Clusters')
        plt.title('Relationship Between predicted Cell Clusters and Categories')
        plt.savefig(os.path.join(figure_dir, f'Cell_Class_label_heatmap_{name}.png'))

def visualize_graph_accuracy(value_dict, IDs, exps, name, figure_dir):
    """
    Visualize Cosine simularity between predicted ROI expression and observed ROI expression.

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    name (str): name of .h5ad file to save scanpy.AnnData, in figure path name
    figure_dir (str): Path to save figures to
    """
    adata_y = get_bulk_expression_of(value_dict, IDs, exps, key='y')
    adata_p = get_bulk_expression_of(value_dict, IDs, exps, key='roi_pred')

    corr_p = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    corr_s = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    corr_k = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    pval_p = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    pval_s = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    pval_k = np.ndarray(adata_p.obs['ID'].unique().shape[0])
    sorted_ids = sorted(adata_p.obs['ID'].unique().tolist())
    for i, id in enumerate(sorted_ids):
        corr_p[i], pval_p[i] = total_corr(adata_p.X[adata_p.obs['ID']==id].flatten(),
                                          adata_y.X[adata_y.obs['ID']==id].flatten(),
                                          method='PEARSONR')
        corr_s[i], pval_s[i] = total_corr(adata_p.X[adata_p.obs['ID']==id].flatten(),
                                          adata_y.X[adata_y.obs['ID']==id].flatten(),
                                          method='SPEARMANR')
        corr_k[i], pval_k[i] = total_corr(adata_p.X[adata_p.obs['ID']==id].flatten(),
                                          adata_y.X[adata_y.obs['ID']==id].flatten(),
                                          method='KENDALLTAU')

    correlation_data = {
        'IDs': sorted_ids,
        'Pearson Correlation Coef.': [corr for corr in corr_p],
        'Pearson p-value': [corr for corr in pval_p],
        'Spearman Correlation Coef.': [corr for corr in corr_s],
        'Spearman p-value': [corr for corr in pval_s],
        'Kendall Correlation Coef.': [corr for corr in corr_k],
        'Kendall p-value': [corr for corr in pval_k]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'IDs': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'IDs': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(figure_dir, f'corr_IDs_{name}.csv'))

    corr_p = np.ndarray(adata_p.obs['files'].unique().shape[0])
    corr_s = np.ndarray(adata_p.obs['files'].unique().shape[0])
    corr_k = np.ndarray(adata_p.obs['files'].unique().shape[0])
    pval_p = np.ndarray(adata_p.obs['files'].unique().shape[0])
    pval_s = np.ndarray(adata_p.obs['files'].unique().shape[0])
    pval_k = np.ndarray(adata_p.obs['files'].unique().shape[0])
    sorted_files = sorted(adata_p.obs['files'].unique().tolist())
    for i, file in enumerate(sorted_files):
        corr_p[i], pval_p[i] = total_corr(adata_p.X[adata_p.obs['files']==file],
                                          adata_y.X[adata_y.obs['files']==file],
                                          method='PEARSONR')
        corr_s[i], pval_s[i] = total_corr(adata_p.X[adata_p.obs['files']==file],
                                          adata_y.X[adata_y.obs['files']==file],
                                          method='SPEARMANR')
        corr_k[i], pval_k[i] = total_corr(adata_p.X[adata_p.obs['files']==file],
                                          adata_y.X[adata_y.obs['files']==file],
                                          method='KENDALLTAU')

    correlation_data = {
        'files': sorted_files,
        'Pearson Correlation Coef.': [corr for corr in corr_p],
        'Pearson p-value': [corr for corr in pval_p],
        'Spearman Correlation Coef.': [corr for corr in corr_s],
        'Spearman p-value': [corr for corr in pval_s],
        'Kendall Correlation Coef.': [corr for corr in corr_k],
        'Kendall p-value': [corr for corr in pval_k]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'files': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'files': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(figure_dir, f'corr_files_{name}.csv'))

    similarity = torch.nn.CosineSimilarity()
    adata_p.obs['cs'] = similarity(torch.from_numpy(adata_p.X), torch.from_numpy(adata_y.X)).squeeze().detach().numpy()

    df = pd.DataFrame({'ID': adata_p.obs['ID'].values,
                  'files': adata_p.obs['files'].values,
                  'cosine_similarity': adata_p.obs['cs'].values})
    mean_value = adata_p.obs['cs'].values.mean()
    mean_row = pd.DataFrame({'ID': 'mean', 'files': 'all', 'cosine_similarity': mean_value}, index=[0])
    df = pd.concat([mean_row, df], ignore_index=True)
    df.to_csv(os.path.join(figure_dir, f'cosine_similarity_{name}.csv')) 

    plt.close('all')
    boxplot = plt.boxplot(adata_p.obs['cs'],)# labels=[category])
    outliers = [flier.get_ydata() for flier in boxplot['fliers']]

    for j, outlier_y in enumerate(outliers):
        outlier_x = np.full_like(outlier_y, 1.1)
        #plt.scatter(outlier_x, outlier_y, marker='o', color='red', label='Outliers' if j == 0 else '')

        for x, y, info in zip(outlier_x, outlier_y, adata_p.obs['files']):
            plt.annotate(info, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')


    plt.ylabel('Cosine Similarity')
    plt.title('Boxplots of Cosine Similarity')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'all_boxplot_{name}.png'))
    plt.close()

    plt.figure(figsize=(30, 5))
    plt.scatter(adata_p.obs['ID'].apply(lambda x: str(x)).values, adata_p.obs['cs'], s=10)
    plt.title('Cosine Similarity of IDs')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.xlabel('IDs')
    plt.savefig(os.path.join(figure_dir, f'cosine_similarity_IDs__{name}.png'))
    plt.close()

    df = pd.DataFrame()
    df['cs'] = adata_p.obs['cs'].values
    df['slides'] = adata_p.obs['files'].apply(lambda x: x.split('-')[-1]).values
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, y='cs', x='slides')
    plt.title('Cosine Similarity of Slides')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Slides')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.savefig(os.path.join(figure_dir, f'cosine_similarity_slides_{name}.png'))
    plt.close()

def visualize_per_gene_corr(value_dict, IDs, exps, name, figure_dir):
    """
    Create tables of gene/protein wise correlation between predicted ROI expression and observed ROI expression.

    Parameters:
    value_dict (dict): dict of file name keys and corresponding ROI expression and predicted sc ROI expression
    IDs (np.array): IDs of sorted value_dict keys
    exps (np.array): Gene/Protein names
    name (str): name of .h5ad file to save scanpy.AnnData, in figure path name
    figure_dir (str): Path to save figures to
    """
    adata_y = get_bulk_expression_of(value_dict, IDs, exps, key='y')
    adata_p = get_bulk_expression_of(value_dict, IDs, exps, key='roi_pred')

    pred = adata_p.X#[adata_p.obs['files'].str.contains('X-').values]       # Filter out some files when no known bulk data
    y = adata_y.X#[adata_p.obs['files'].str.contains('X-').values]

    p_statistic, p_pval = per_gene_corr(pred, y, mean=False, method='PEARSONR')
    s_statistic, s_pval = per_gene_corr(pred, y, mean=False, method='SPEARMANR')
    k_statistic, k_pval = per_gene_corr(pred, y, mean=False, method='KENDALLTAU')

    mi = per_gene_mi(pred, y)
        
    correlation_data = {
        'Variable': adata_p.var_names.values,
        'Pearson Correlation Coef.': [corr for corr in p_statistic],
        'Pearson p-value': [corr for corr in p_pval],
        'Spearman Correlation Coef.': [corr for corr in s_statistic],
        'Spearman p-value': [corr for corr in s_pval],
        'Kendall Correlation Coef.': [corr for corr in k_statistic],
        'Kendall p-value': [corr for corr in k_pval],
        'mi': [corr for corr in mi],
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(figure_dir, f'corr_area_{name}.csv'))

    correlation_data = {
        'Variable': adata_p.var_names.values[p_pval < 0.05],
        'Pearson Correlation Coef.': [p_statistic[i] for i in range(p_statistic.shape[0]) if p_pval[i] < 0.05],
        'Pearson p-value': [p_pval[i] for i in range(p_pval.shape[0]) if p_pval[i] < 0.05]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(figure_dir, f'corr_area_pearson_filter_{name}.csv'))

    correlation_data = {
        'Variable': adata_p.var_names.values[s_pval < 0.05],
        'Spearman Correlation Coef.': [s_statistic[i] for i in range(s_statistic.shape[0]) if s_pval[i] < 0.05],
        'Spearman p-value': [s_pval[i] for i in range(s_pval.shape[0]) if s_pval[i] < 0.05]
    }

    corr_df = pd.DataFrame(correlation_data)
    mean_values = corr_df[corr_df.columns[1:]].mean()
    mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
    std_values = corr_df[corr_df.columns[1:]].std()
    std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
    corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

    corr_df.to_csv(os.path.join(figure_dir, f'corr_area_spearman_filter_{name}.csv'))
    return adata_p.var_names.values[s_pval < 0.05]

def visualizeExpression(processed_dir='TMA1_processed',
                        embed_dir='out/',
                        label_data='label_data.csv',
                        figure_dir='figures/',
                        name='_cells',
                        select_cells=50000):
    """
    processed_dir (str): Dir name in data/processed/ containing torch_geometric graphs
    embed_dir (str): Path to dir containing predicted sc expression
    label_data (str): Name of .csv in data/raw/ containing label information of ROIs(IDs specificly)
    figure_dir (str): Path to save figures to
    name (str): name of .h5ad file to save/load scanpy.AnnData, in figure path name
    select_cells (int): Number of cells to analyse, if 0 select all, otherwise random specified subset
    """
    value_dict = get_true_graph_expression_dict(os.path.join('data/processed', processed_dir))
    value_dict = get_predicted_graph_expression(value_dict, embed_dir)
    value_dict, cell_shapes = get_predicted_cell_expression(value_dict, embed_dir)
    IDs, exps = get_patient_ids(label_data, list(value_dict.keys()))
    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    # visualize_bulk_expression(value_dict, IDs, exps, '_true', key='y')
    # visualize_bulk_expression(value_dict, IDs, exps, '_pred', key='roi_pred')
    visualize_graph_accuracy(value_dict, IDs, exps, name, figure_dir)
    spearman_genes = visualize_per_gene_corr(value_dict, IDs, exps, name, figure_dir)
    visualize_cell_expression(value_dict, IDs, exps, name, figure_dir, cell_shapes, spearman_genes, select_cells)
    
