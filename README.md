# Transient Hotspots

## 0. Local copy

Clone the repository and `cd` to the dir.
```
git clone https://github.com/ThibaultLatrille/TransientHotspots
cd TransientHotspots
```

## 1. Installation

Install python3 packages
```
sudo apt install -qq -y snakemake python3-pandas python3-pip
```

## 2. Run snakemake

Run `snakemake` with 4 CPU and parameters given by `config.yaml`: 
```
snakemake --cores 4 
```

## Licence

The MIT License (MIT)

Copyright (c) 2021 Thibault Latrille & Julien Joseph

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


