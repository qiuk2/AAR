from .SAT import SAT
from .msstftd import MultiScaleSTFTDiscriminator
from .audio_to_mel import Audio2Mel
from .aar import AAR


def build_aar(
    vae: SAT, depth: int,
    input_dim: int=512,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    aln=1, aln_gamma_init=1e-3, shared_aln=False, layer_scale=-1,
    tau=4, cos_attn=False,
    flash_if_available=True, fused_if_available=True,
):
    aar = AAR(
        vae_local=vae, input_dim=input_dim, patch_nums=patch_nums,
        depth=depth, embed_dim=depth*64, num_heads=depth, drop_path_rate=0.1 * depth/24,
        aln=aln, aln_gamma_init=aln_gamma_init, shared_aln=shared_aln, layer_scale=layer_scale,
        tau=tau, cos_attn=cos_attn,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )
    aar.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)
    return aar