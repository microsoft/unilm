from multihead_diffattn import MultiheadDiffAttn
from multihead_attention import MultiheadAttention

if __name__ == "__main__":
    # Diff Attention with MHA, 1024 embed_dim, 8 heads, 8 kv_heads
    diff_attn_mha = MultiheadDiffAttn(embed_dim=1024, depth=0, num_heads=8, num_kv_heads=None)
    # can be compared against baseline Attention with MHA, 1024 embed_dim, 16 heads, 16 kv_heads
    attn_mha = MultiheadAttention(embed_dim=1024, depth=0, num_heads=16, num_kv_heads=None)
    # write code to print their number of parameters
    print("Number of parameters in 1 layer diff_attn_mha:", sum(p.numel() for p in diff_attn_mha.parameters()))
    print("Number of parameters in 1 layer attn_mha:", sum(p.numel() for p in attn_mha.parameters()))


    # Diff Attention with GQA, 1024 embed_dim, 8 heads, 4 kv_heads
    diff_attn_gqa = MultiheadDiffAttn(embed_dim=1024, depth=0, num_heads=8, num_kv_heads=4)
    # can be compared against baseline Attention with GQA, 1024 embed_dim, 16 heads, 8 kv_heads
    attn_gqa = MultiheadAttention(embed_dim=1024, depth=0, num_heads=16, num_kv_heads=8)
    print("Number of parameters in 1 layer diff_attn_gqa:", sum(p.numel() for p in diff_attn_gqa.parameters()))
    print("Number of parameters in 1 layer attn_gqa:", sum(p.numel() for p in attn_gqa.parameters()))
