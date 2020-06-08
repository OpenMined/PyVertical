![om-logo](https://github.com/OpenMined/design-assets/blob/master/logos/OM/horizontal-primary-trans.png)

![License](https://img.shields.io/github/license/OpenMined/PyVertical)
![OpenCollective](https://img.shields.io/opencollective/all/openmined)

# PyVertical

A project developing Privacy Preserving Vertically Distributed Learning.

- :lock: Links vertically partitioned data
         without exposing membership
         using Private Set Intersection (PSI)
- :eye: Trains a model on vertically partitioned data
        using SplitNNs,
        so only data holders can access data

## Requirements
This project is written in Python.
The work is displayed in jupyter notebooks.

To install the dependencies,
we recommend using Conda:
1. Clone this repository
1. In the command line, navigate to your local copy of the repository
1. Run `conda env create -f environment.yml`
    - This creates an environment `pyvertical-dev`
    - Comes with most dependencies you will need
1. Activate the environment with `conda activate pyvertical-dev`
1. Run `pip install syft[udacity]`
1. Run `conda install notebook`

N.b. Installing the dependencies takes several steps to circumvent versioning incompatibility between
`syft` and `jupyter`.
In the future,
all packages will be moved into the `environment.yml`.

## Contributing
Pull requests are welcome.
For major changes,
please open an issue first to discuss what you would like to change.

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)