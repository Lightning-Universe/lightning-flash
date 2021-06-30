# Installation

Flash is tested on Python 3.6+, and PyTorch 1.6.

## Install with pip

```bash
pip install lightning-flash
```

Optionally, you can install Flash with extra packages for each domain or all domains.
```bash
pip install 'lightning-flash[image]'
pip install 'lightning-flash[tabular]'
pip install 'lightnign-flash[text]'
pip install 'lightning-flash[video]'

# image + video
pip install 'lightning-flash[vision]'

# all features
pip install 'lightning-flash[all]'
```

For contributors, please install Flash with packages for testing Flash and building docs.
```bash
# Clone Flash repository locally
git clone https://github.com/[your username]/lightning-flash.git
cd lightning-flash

# Install Flash in editable mode with extra packages for development
pip install -e '.[dev]'
```

## Install from source

```bash
pip install git+https://github.com/PyTorchLightning/lightning-flash.git
```
