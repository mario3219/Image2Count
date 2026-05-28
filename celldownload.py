import os
import argparse

"""

Model weights for EVA has to be downloaded from hugging face
    https://huggingface.co/yandrewl/Eva
and renamed to:
    eva_model.pt
and moved to:
    out/models

And GenePT marker embeddings from
    https://zenodo.org/records/10833191
Use the file:
    GenePT_gene_protein_embedding_model_3_text.pickle
and store it as:
    GenePT_embedding.pkl
in src/utils/eva_kit

KRONOS requires marker_metadata.csv from huggingface
    https://huggingface.co/MahmoodLab/KRONOS
Download and place in:
src/utils/kronos_kit

"""

def parse_args():
    parser = argparse.ArgumentParser(description=
                                    """
                                    Script for downloading image encoders from foundation models.
                                    """)
    parser.add_argument("--model",type=str,default="",
                        help="""
                        Name of foundation model to download, available options are:
                        deepcell, kronos
                        """)
    return parser.parse_args()

def main(**args):
    
    model_path = os.path.join(os.getcwd(),"out","models")
    if not os.path.exists():
        os.makedirs(model_path)

    if args["model"] == "deepcell":
        from src.utils.download_utils.dct_download import dct_download
        dct_download(model_path)
    if args["model"] == "kronos":
        from src.utils.download_utils.kr_download import kr_download
        kr_download(model_path)

if __name__=="__main__":
    args = vars(parse_args())
    main(**args)
