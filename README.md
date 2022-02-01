# Inspectro

Spectral decomposition for characterizing long-range interaction profiles in Hi-C maps. 

Snakemake workflow for the unsupervised method presented in [Spracklin, Abdennur et al., 2021](https://www.biorxiv.org/content/10.1101/2021.08.05.455340v1).


### Steps
0. Set up your environment with [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba](https://github.com/mamba-org/mamba):

```sh
conda env create -n spec -f environment.yml
conda activate spec
```

1. Place supplementary bigwig file information in `config/track_metadata.tsv` to include in graphical outputs. Must be tab-delimited, including header. Must have columns:

* `Name`: a display name
* `ID`: a unique identifier to use in the database (can be the same as `Name`)
* `FileFormat`: must be the string `bigWig`
* `Path`: a local path to the file

2. Edit `config.yaml`.

3. Run:

```sh
# Generate pre-aggregated supplementary bigwig tracks if provided.
$ snakemake make_track_db --force --cores all

# Run the pipeline
$ snakemake --cores all
```
