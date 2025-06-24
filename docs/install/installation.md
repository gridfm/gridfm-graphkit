You can install `gridfm-graphkit` directly from PyPI:

```bash
pip install gridfm-graphkit
```

---

## Development Setup

To contribute or develop locally, clone the repository and install in editable mode:

```bash
git clone git@github.com:gridfm/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install -e .
```

For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install -e .[dev,test]
```
