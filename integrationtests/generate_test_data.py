import urllib.request
import yaml
import subprocess


def execute_and_live_output(cmd) -> None:
    subprocess.run(cmd, text=True, shell=True, check=True)


def _base_config() -> dict:
    """
    Download the default config from gridfm-datakit and apply common test parameters.
    """
    config_url = (
        "https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main"
        "/scripts/config/default.yaml"
    )

    print(f"Downloading config from {config_url}...")
    with urllib.request.urlopen(config_url) as response:
        config_content = response.read().decode("utf-8")

    config = yaml.safe_load(config_content)

    config["network"]["name"] = "case14_ieee"
    config["load"]["scenarios"] = 10000
    config["topology_perturbation"]["n_topology_variants"] = 2

    return config


def generate_pf_test_data(config_path: str = "integrationtests/default_pf.yaml") -> None:
    """
    Generate power-flow (PF) test data for case14_ieee with 10 000 scenarios
    and 2 topology variants.
    """
    config = _base_config()

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"PF config written to {config_path}")
    print(f"  network.name                          : {config['network']['name']}")
    print(f"  load.scenarios                        : {config['load']['scenarios']}")
    print(f"  topology_perturbation.n_topology_variants: {config['topology_perturbation']['n_topology_variants']}")

    execute_and_live_output(f"gridfm_datakit generate {config_path}")


def generate_opf_test_data(config_path: str = "integrationtests/default_opf.yaml") -> None:
    """
    Generate optimal power-flow (OPF) test data for case14_ieee with 10 000 scenarios
    and 2 topology variants.
    """
    config = _base_config()
    config["settings"]["mode"] = "opf"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"OPF config written to {config_path}")
    print(f"  network.name                          : {config['network']['name']}")
    print(f"  load.scenarios                        : {config['load']['scenarios']}")
    print(f"  topology_perturbation.n_topology_variants: {config['topology_perturbation']['n_topology_variants']}")
    print(f"  settings.mode                         : {config['settings']['mode']}")

    execute_and_live_output(f"gridfm_datakit generate {config_path}")


if __name__ == "__main__":
    #generate_pf_test_data()
    generate_opf_test_data()
