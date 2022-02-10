# Installation & Troubleshooting

## Installation options

### Install with pip

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

### Install with conda

Flash is available via conda forge. Install it with:
```bash
conda install -c conda-forge lightning-flash
```

### Install from source

You can install Flash from source without any domain specific dependencies with:
```bash
pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git'
```

To install Flash with domain dependencies, use:
```bash
pip install 'git+https://github.com/PyTorchLightning/lightning-flash.git#egg=lightning-flash[image]'
```

You can again install dependencies for multiple domains by separating them with commas as above.


## Troubleshooting

### Torchtext incompatibility

If you install Flash in an environment that already has a version of torchtext installed, you may see an error like this when you try to import it:

```bash
ImportError: /usr/local/lib/python3.7/dist-packages/torchtext/_torchtext.so: undefined symbol: _ZN2at6detail10noopDeleteEPv
```

The workaround is to uninstall torchtext __before__ installing Flash, like this:

```bash
pip uninstall -y torchtext
pip install lightning-flash[...]
```

### FiftyOne incompatibility on Google Colab

When installing Flash (or PyTorch Lightning) alongside FiftyOne in a Google Colab environment, you may get the following error when importing FiftyOne:

```bash
ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port
```

There is no known workaround for this issue at the time of writing, but you can view the latest updates on the [associated github issue](https://github.com/voxel51/fiftyone/issues/1376).
