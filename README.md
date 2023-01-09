# Inspectro

Spectral decomposition for characterizing long-range interaction profiles in Hi-C maps. 

Snakemake workflow for the unsupervised method presented in [Spracklin, Abdennur et al., 2022](https://www.nature.com/articles/s41594-022-00892-7).

```bibtex
@article{spracklin2022diverse,
  title={Diverse silent chromatin states modulate genome compartmentalization and loop extrusion barriers},
  author={Spracklin, George and Abdennur, Nezar and Imakaev, Maxim and Chowdhury, Neil and Pradhan, Sriharsa and Mirny, Leonid A and Dekker, Job},
  journal={Nature Structural \& Molecular Biology},
  pages={1--14},
  year={2022},
  publisher={Nature Publishing Group}
}
```


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
