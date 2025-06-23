Create a python virtual environment and install the requirements
```bash
git clone git@github.com:gridfm/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
pip install .
```

Install the package in editable mode during development phase:

```bash
pip install -e .[dev,test]
```
