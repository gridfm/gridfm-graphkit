import torch
import numpy as np
import pytest
import yaml
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask

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
        "v0_1",
        "examples/models/GridFM_v0_1.pth",
        "tests/data/reference_output_v0_1.npy",
        "examples/config/gridFMv0.1_pretraining.yaml",
    ),
    (
        "v0_2",
        "examples/models/GridFM_v0_2.pth",
        "tests/data/reference_output_v0_2.npy",
        "examples/config/gridFMv0.2_pretraining.yaml",
    ),
]


@pytest.mark.parametrize(
    "version, model_path, ref_output_path, yaml_path",
    models_to_test,
)
def test_model_matches_reference(version, model_path, ref_output_path, yaml_path):
    torch.manual_seed(0)
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)

    # Prepare zero input
    x = torch.zeros((num_nodes, x_dim))
    pe = torch.zeros((num_nodes, pe_dim))
    edge_index = torch.tensor([[0], [0]])
    edge_attr = torch.zeros((1, edge_attr_dim))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # load model
    model = FeatureReconstructionTask(args, None, None)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
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
