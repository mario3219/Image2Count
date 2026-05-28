def DeepCell(
        n_filters=256,
        n_heads=4,
        n_domains=9,
        embed_size=256,
        **args
        ):

    from src.utils.deepcell_kit.config import DCTConfig
    import numpy as np
    from src.models.DeepCellModel import CellTypeDataEncoder

    dct_config = DCTConfig()

    embedding_model_name = "deepseek-r1-70b-llama-distill-q4_K_M"
    
    marker2embedding = dct_config.get_channel_embedding(
        embedding_model_name=embedding_model_name
    )

    marker_embeddings = np.zeros_like(list(marker2embedding.values()), dtype=np.float32)
    for marker, ebd in marker2embedding.items():
        if marker not in dct_config.marker2idx:
            print("bad_marker?", marker)
        idx = dct_config.marker2idx[marker]
        marker_embeddings[idx] = ebd

    model = CellTypeDataEncoder(
        n_filters=n_filters, 
        n_heads=n_heads,
        embed_size=256,
        marker_embeddings=marker_embeddings,
        img_feature_extractor='conv',
    )
    return model

def KRONOS(patch_size=16, num_register_tokens=0,**kwargs):
    from src.models.KRONOSModel import DinoVisionTransformer
    from src.utils.kronos_kit.block import NestedTensorBlock as Block
    from src.utils.kronos_kit.attention import MemEffAttention
    from functools import partial
    
    model = DinoVisionTransformer(
        patch_size=patch_size,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
