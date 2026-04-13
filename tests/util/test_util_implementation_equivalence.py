import torch
import pytest


def impl_a(P_in, Pd, p_shunt, agg_bus, mask_pv, mask_ref):
    Pg_new = torch.zeros_like(P_in)
    Pg_new[mask_pv] = agg_bus[mask_pv]
    Pg_new[mask_ref] = P_in[mask_ref] + Pd[mask_ref] - p_shunt[mask_ref]
    return Pg_new


def impl_b(P_in, Pd, p_shunt, agg_bus, mask_pv, mask_ref):
    Pg_ref = torch.where(mask_ref, P_in + Pd - p_shunt, torch.zeros_like(P_in))
    Pg_new = torch.where(mask_pv, agg_bus, Pg_ref)
    return Pg_new


@pytest.mark.parametrize("allow_overlap", [False, True])
def test_run(allow_overlap):
    n = 10000
    device = "cpu"
    torch.manual_seed(0)

    P_in = torch.randn(n, device=device)
    Pd = torch.randn(n, device=device)
    p_shunt = torch.randn(n, device=device)
    agg_bus = torch.randn(n, device=device)

    # random masks
    mask_pv = torch.rand(n, device=device) > 0.7
    mask_ref = torch.rand(n, device=device) > 0.7

    if not allow_overlap:
        mask_ref = mask_ref & (~mask_pv)  # enforce disjointness

    out_a = impl_a(P_in, Pd, p_shunt, agg_bus, mask_pv, mask_ref)
    out_b = impl_b(P_in, Pd, p_shunt, agg_bus, mask_pv, mask_ref)

    equal = torch.allclose(out_a, out_b)
    max_diff = (out_a - out_b).abs().max().item()

    print(equal, max_diff)
    if not allow_overlap:
        assert equal, f"Outputs differ! Max abs diff: {max_diff:.6f}"
    else:
        assert max_diff > 0, (
            "Outputs are identical even with overlapping masks, which is unexpected!"
        )
        assert not equal, (
            "Outputs are identical despite overlapping masks, which is unexpected!"
        )
