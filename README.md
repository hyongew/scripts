# Scripts catalog

Small utility scripts. Some vibe-coded, some not.

## `convert_image.py`

Converts an image to other formats (currently only does PNG to JPEG)

```bash
python convert_image.py input.png output.jpg
```

requirements: `pip install pillow`.

## `rename.py`

Renames and organises pictures (but can be used for any file).

```bash
python rename.py
```

## `llama-fit.py`

Get fit params for llama.cpp, and runs speed benchmarks.

```bash
python llama-fit.py
```

## `pillama.ps1`

PowerShell script to serve llama.cpp and launches pi connecting to it.

```powershell
$env:PILLAMA_MODELS_DIR = 'C:\Path\To\GGUFs\'
.\pillama.ps1
```

The script expects `llama` and `pi` to be available on `PATH`.
