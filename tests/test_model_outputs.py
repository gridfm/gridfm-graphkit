import torch
import numpy as np
import pytest

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input shape config
num_nodes = 1
x_dim = 9
pe_dim = 20
edge_attr_dim = 2

# List of models and reference files to check
models_to_test = [
    (
        "v0_1_2",
        "examples/models/GridFM_v0_1_2.pth",
        "tests/data/reference_output_v0_1_2.npy",
    ),
    (
        "v0_2_3",
        "examples/models/GridFM_v0_2_3.pth",
        "tests/data/reference_output_v0_2_3.npy",
    ),
]


@pytest.mark.parametrize("version, model_path, ref_output_path", models_to_test)
def test_model_matches_reference(version, model_path, ref_output_path):
    torch.manual_seed(0)

    # Prepare zero input
    x = torch.zeros((num_nodes, x_dim), device=device)
    pe = torch.zeros((num_nodes, pe_dim), device=device)
    edge_index = torch.tensor([[0], [0]], device=device)
    edge_attr = torch.zeros((1, edge_attr_dim), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Load model
    model = torch.load(model_path, weights_only=False, map_location=device).to(device)
    model.eval()

    # Get current output
    with torch.no_grad():
        output = model(x, pe, edge_index, edge_attr, batch).cpu().numpy()

    # Load saved reference
    reference = np.load(ref_output_path)

    # Exact match assertion
    assert np.allclose(output, reference, rtol=1e-5, atol=1e-6), (
        f"Model output for {version} does not match reference within tolerance.\n"
        f"Max absolute difference: {np.max(np.abs(output - reference))}"
    )
