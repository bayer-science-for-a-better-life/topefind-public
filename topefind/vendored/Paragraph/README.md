# Paragraph

**Antibody paratope prediction using Graph Neural Networks with minimal feature vectors.**

The development of new vaccines and antibody therapeutics typically takes several years and requires over $1bn in investment. Accurate knowledge of the paratope (antibody binding site) can speed up and reduce the cost of this process by improving our understanding of antibody-antigen binding.


## Install

To download and install the latest version from github:

```
# clone the repo
git clone https://github.com/oxpig/Paragraph.git
cd Paragraph/

# create your virtual env e.g.
python3 -m venv paragraph_pip_env
source paragraph_pip_env/bin/activate

# install
pip install .
```

If you are having issues installing, try upgrading pip: *pip install --upgrade pip*.

Paragraph uses Python v3.8.3.


## Stats

Paragraph's predictions take less than 0.1s per structure. 10,000 structures take approximately 10 minutes to process and the resulting output file is ~50MB.


## Usage

To predict the paratopes of your crystal or model antibody structures using the command line, use a command similar to the below.

```
Paragraph --pdb_H_L_csv     /your/abspath/to/Paragraph/Paragraph/example/pdb_H_L_key.csv
          --pdb_folder_path /your/abspath/to/Paragraph/Paragraph/example/pdbs
          --out_path        /your/abspath/to/desired/save/location/of/predictions.csv
```

When trained on paired data, it is expected that Paragraph learns information pertinent to both the heavy and light chains. To ensure Paragraph's viability when only one chain is available, Paragraph has also been trained on only heavy and only light chains. These single chain weights can be used with the appropriate flag.

```
Paragraph --pdb_H_L_csv     /your/abspath/to/Paragraph/Paragraph/example/pdb_H_L_key.csv  # with L ids removed
          --pdb_folder_path /your/abspath/to/Paragraph/Paragraph/example/pdbs
          --out_path        /your/abspath/to/desired/save/location/of/predictions.csv
          --heavy
```

An example jupyter notebook is provided for those wishing to integrate Paragraph into their python workflow. Examples of correctly formatted input data are also provided. PDB files should be IMGT numbered.

## Output

Paragraph outputs a csv file containing predicted probabilities for each residue belonging to the paratope. The output csv is formatted as below. The atom number and 3D coordinates are of the C-alpha atoms for each residue.

In order to recapitulate the true number of binding residues observed in the CDR loops plus two extra residues on either end, we recommend using a classifier cut-off of 0.734 on Paragraph's predictions.

<div align="center">

| pdb | chain_type | chain_id | IMGT | AA | atom_num | x | y | z | pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4edw | H | H | 25 | VAL | 1011  | 9.294 | -11.476 | -36.290 | 0.009999541 |
| 4edw | H | H | 26 | SER | 1018 | 12.006 | -13.600 | -38.105 | 0.010073511
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

</div>


## Citation

```
@article{Chinery2022,
  title={Paragraph - Antibody paratope prediction using Graph Neural Networks with minimal feature vectors},
  author={Lewis Chinery, Newton Wahome, Iain H. Moal, and Charlotte M. Deane},
  journal={bioRxiv},
  year={2022}
}
```
