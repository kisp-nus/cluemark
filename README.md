# CLUE-Mark
****Watermarking diffusion model outputs using CLWE****

This is the proof-of-concept of the CLUE-Mark system for watermarking diffusion model images. For theory and background on CLUE-Mark's construction, see our [paper](https://maps.kisp-lab.org/cluemark/).

## Setup

CLUE-Mark requires PyTorch with the associated NVidia drivers, NumPy, MatPlotLib, and a variety of other supporting libraries. It is by far easiest to configure by installing [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) and then using the included package list to configure a fresh environment:

```bash
conda env create -n cluemark -f conda_env.yml
conda activate cluemark
```

## Quickstart

To reproduce the results from the paper, it is easiest to simply invoke the runner script. First, create the unwatermarked images.

```bash
./run_wm_checks.sh -b base cuda:0
```

Then invoke a watermark and check the results.

```bash
./run_wm_checks.sh cluemark cuda:0
```

Configurations for `tree_ring` and `gaussian_shading` are also included for comparison. The default script performs the following:

1. Generates watermarked images, which are placed in the `images/` folder.
2. Checks the FID score of the images against the unwatermarked images.
3. Checks for recovery of the watermark and compares the watermark scores for unwatermarked images. Output is written to `results/`. Filters are also applied to check the performance of watermarks after perturbations.
4. Applies a steganographic attack to try to remove the watermark and reports resulting recovery scores.

Options:
* `-n` skips image generation to just do steps 2 to 4 above.
* `-s` allows you to set a starting image number.
* `-e` allows you to set an ending image number.
* `-d` allows you to set the dataset. Default is `sdp` (Stable Diffusion Prompts), `coco` is also supported.

To get AUC scores from output files use the `auc_from_results.py` file. Example:

```bash
python auc_from_results.py results/sdp/enh_inv_dpm/cluemark.txt
```

Note that depending on the orientation of the scoring, 0 or 1 might indicate success, while 0.5 is always worst. For comparison, you may need to invert AUC scores from some watermarks.

## Outputs

Unless configured otherwise, images are output to the `images/` folder while all other output data is written to `results/`. In both cases, output is grouped into subfolders by dataset, solver, and experiment. For example, the watermark detection data for CLUE-Mark on Stable Diffusion Prompts using the enhanced inverse DPM solver will be output to `results/sdp/enh_inv_dpm/cluemark.txt`, while output for Gaussian Shading using DDIM on the coco dataset is written to `results/coco/ddim/gaussian_shading.txt`.

****All results data is append-only in CSV format.**** If you restart the pipeline or run it twice, you will end up with a text file with two sets of data, including a double-header. After which, if you try to run the AUC script you will get an error similar to `Input contains NaN`. This is intentional: it forces you to check that the file is consistent before running the AUC script, while avoiding any possible data loss. To fix this, remove the extra header and make sure the data is correct before running the AUC script.

A quick description of the output files:

* The `[exp_name].txt` file contains the watermark scores for the various filters configured with and without watermarks.
* The `[exp_name]-steg.txt` file contains the watermark scores after applying a steganographic attack vs the unwatermarked image.
* The `[exp_name]-fid.txt` file contains the FID distance of *all* the watermarked images vs the unwatermarked images in the given folder. This distance is meant to be a measure of quality, but is really a measure of similarity between the watermarked and unwatermarked images, with lower scores meaning more similar.

For the first two files, using the `auc_from_results.py` script will give you the Area Under the Curve (AUC) of the Receiver Operating Characteristic for the watermark resutls, with 1.0 indicating perfect distinguishing of watermarked and unwatermarked cases, while 0.5 is equivalent to random guessing. A score less than 0.5 indicates that the labels are flipped (which occurs in some configurations, simply subtract from 1 to compare to others). See the paper for more details on how to interpret these results.

## Running Scripts

All scripts are configured via [YAML](https://yaml.org) files, which are always the first command line arguments. This allows for tight control and reproduction of experiments across scripts. See the [Configuration](#Configuration) section below for configuration details. Example script invocation:

```bash
python generate_images.py config/cluemark.yaml start=0 end=100 device=cuda:0
```

Available scripts:
* `generate_images.py` Generates watermarked images using the configured watermark and pipeline using prompts from the configured dataset.
* `check_watermark.py` Applies configured filters to both the watermarked and non-watermarked images and scores both.
* `score_images.py` applies FID to score distance of watermarked images to the unwatermarked images. NB: applies to the entire folder, ignores start and end options.
* `steg_from_images.py` applies a steganographic attack to the watermarked images, then checks for watermark recovery.
* `auc_from_results.py` does ***not*** take a configuration file, but rather the results CSV for watermark scores and computes AUC values.

## Configuration

All scripts are configured via [YAML](https://yaml.org) files, which are always the first command line arguments. This allows for tight control and reproduction of experiments across scripts. Different YAML sections control different stages of the pipeline, and configuration files can import and override base configurations to produce a cascade. Configurations can also be overriden on the command line and are seamlessly merged. For example, if the `config/base.yaml` file specifies:

```yaml
dataset: sdp
seed: 42
device: ???
exp_name: "no_wm"
```

And `config/my_exp.yaml` specifies:

```yaml
include: [
  "config/base.yaml",
  exp_name: "my_exp"
]
```

And then you run `some_script.py config/my_exp.yaml dataset=coco device=cuda:1` then the resulting configuration will be:

```yaml
dataset: coco
seed: 42
device: cuda:1
exp_name: "my_exp"
```

## Global Options

Here are the most useful options. There are a variety of intermediate variables used for grouping and allowing overrides, but those generally should not be modified unless you really need to.

| Option | Default | Notes |
|--------|---------|-------|
| device | | Which CUDA device to set for PyTorch. Mandatory. |
| dataset | sdp | Dataset to use for model inputs. Default is Stable Difussion Prompts. Coco is also supported as `coco`. |
| seed | 42 | Seed to apply to random number generators to ensure reproducibility of results. |
| exp_name | no_wm | Name to be used for folders and output results. |
| images_folder | ./images | Folder for images produced by watermarking scheme. |
| results_folder | ./results | Folder for writing output data, e.g. scoring results. |
| solver_name | "enh_inv_dpm" | Results are typically grouped by different solvers. This should only be set in the base and not in any derived configurations. |
| start | | Starting input or image number, used by many scripts. |
| end | | Ending input or image number, used by many scripts. |


## Pipeline

This is generally configured to be Stable Diffusion 2.1, but allows for configuration if needed. See `pipeline_runner.py` for details of how configurations are used or to extend to other models.

The `solver_order` option controls whether DDIM (0) or Enhanced Inverse DPM (1 or 2) is used for inverting the model. Make sure to set `dtype` to `float32` for order 1 or 2.

## Filters

A variety of filters are available for testing perturbations. These are given as a list of dictionaries, with the `type` field set as the filter name and the rest as required for the given filer:
* `none` passthrough, marked as "clean" in outputs.
* `jpeg` applies JPEG compression with the configured `quality` value.
* `rotate` applies a random rotation up to the configured `degrees` value.
* `crop` applies a random resize and crop up to the configured `scale` and `crop` values.
* `brightness` applies a random brightness change up to the configured `factor`.
* `blur` applies a gaussian blur of the given `radius`.

## Watermarks

Watermarks are configured by setting the `watermark` section and using the `type` field to set the name of the watermark. The rest of the fields are set as required for the watermark.

## CLUE-Mark

| Field | Example | Notes |
|-------|---------|-------|
| type | clwe | |
| secret_dim | `[2, 4, 4]` | Specifies the size and shape of the CLWE secret. Secret is chosen based on the seed. |
| gamma | 2.0 | CLWE $\gamma$ value. |
| beta | 0.001 | CLWE $\beta$ value. |
| dwt_bands | ["LL", "LH", "HL", "HH"] | Applies CLUE-Mark in the DWT (Discrete Wavelet Transform) domain in the specified bands. |
| seed | 54321 | Seed used for both the secret an RNG initialization. |
| dct | True | Apply CLUE-Mark in the Discrete Cosine Transform (DCT) domain. |

## Tree Ring

To apply a Tree Ring watermark, specify `type: tree_ring` in the `watermark` section. See Tree Ring's documentation for configuration (names should match their command line paramters), or the `tree_ring/tree_ring_watermark.py` file.

## Gaussian Shading

To apply a Gaussian Shading watermark, specify `type: gaussian_shading` in the `watermark` section. See Gaussian Shading's documentation for configuration, or the `gaussian_shading/gaussian_shading.py` file for details.

## Notebooks

The `notebooks` folder includes several Jupyter notebooks for performing the statistical simulations used to generate the figures in the paper. The same Anaconda environment should allow loading the notebooks in Jupyter (via web interface or using VSCode) to reproduce the other results in the paper.
