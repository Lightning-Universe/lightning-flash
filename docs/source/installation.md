# Installation

## Install with pip

```bash
pip install lightning-flash
```

Optionally, you can install Flash with extra packages for each domain.

For a single domain, use: `pip install 'lightning-flash[{DOMAIN}]'`.
```bash
pip install 'lightning-flash[image]'
pip install 'lightning-flash[tabular]'
pip install 'lightning-flash[text]'
...
```

For muliple domains, use: `pip install 'lightning-flash[{DOMAIN_1, DOMAIN_2, ...}]'`.
```bash
pip install 'lightning-flash[audio,image]'
...
```

For contributors, please install Flash with packages for testing Flash and building docs.
```bash
# Clone Flash repository locally
git clone https://github.com/[your username]/lightning-flash.git
cd lightning-flash

# Install Flash in editable mode with extra packages for development
pip install -e '.[dev]'
```

## Install with conda

Flash is available via conda forge. Install it with:
```bash
conda install -c conda-forge lightning-flash
```

## Install from source

You can install Flash from source without any domain specific dependencies with:
```bash
pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git'
```

To install Flash with domain dependencies, use:
```bash
pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[image]'
```

You can again install dependencies for multiple domains by separating them with commas as above.


## Installation Issues 

As of September 26th, 2021 there is a known issue with some packages used by Flash that cause installation issues. If you experience an issue installing Flash, see  [Issue 803](https://github.com/PyTorchLightning/lightning-flash/issues/803) for potential workarounds. 
