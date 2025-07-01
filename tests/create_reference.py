# save_reference_output.py
import torch
import numpy as np

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 1
x_dim = 9
pe_dim = 20
edge_attr_dim = 2

# Dummy all-zero input
x = torch.zeros((num_nodes, x_dim)).to(device)
pe = torch.zeros((num_nodes, pe_dim)).to(device)
edge_index = torch.tensor([[0], [0]]).to(device)
edge_attr = torch.zeros((1, edge_attr_dim)).to(device)
batch = torch.zeros(num_nodes, dtype=torch.long).to(device)


models = {
    "v0_1_2": "../examples/models/GridFM_v0_1_2.pth",
    "v0_2_3": "../examples/models/GridFM_v0_2_3.pth",
}

for version, path in models.items():
    print(f"Loading model {version}...")
    model = torch.load(path, weights_only=False, map_location=device).to(device)
    model.eval()

    with torch.no_grad():
        output = model(x, pe, edge_index, edge_attr, batch)

    out_path = f"./data/reference_output_{version}.npy"
    np.save(out_path, output.cpu().numpy())
    print(f"Saved output for {version} to {out_path}")
