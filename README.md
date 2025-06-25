# Code Availability for "Generalized queueing-based evolutionary dynamics rescues cooperation in social dilemmas."

## Project Introduction

This project contains the open-source code implementation of the paper *"Generalized queueing-based evolutionary dynamics rescues cooperation in social dilemmas."*  
The paper proposes a generalized evolutionary dynamics model based on queueing theory, focusing on how naturally occurring "quiescent behavior" affects the evolution of cooperation in social dilemmas when individuals' strategy is limited and information adoption experiences delays.  
Through theoretical analysis and numerical simulations, the paper reveals how quiescent behavior promotes cooperation as the dominant strategy in homogeneous populations, while having less influence on heterogeneous populations.  
This code implements all key numerical simulations and data generation experiments from the paper, enabling readers to reproduce and verify the results.

## Environment Requirements

To ensure the proper operation of the project and reproducibility of results, it is recommended to run the code in the following software environment:

- **Python Version:** ≥ 3.8  
  Python versions 3.8 to 3.11 are recommended.

- **Core Dependencies:**
  - [`networkx`](https://networkx.org/): For constructing and manipulating complex network structures  
  - [`numpy`](https://numpy.org/): For efficient numerical computation and array operations  
  - [`matplotlib`](https://matplotlib.org/): For plotting and data visualization (the main tool used for graphing in this project)

### Installation

It is recommended to install all dependencies via:

```bash
pip install -r requirements.txt
````

(assuming a `requirements.txt` file exists in the project root directory).

## Usage Instructions

To ensure reproducibility and completeness of all figures in the paper, this project provides **independent executable scripts for each figure**. Each script corresponds to one figure and is named in order of appearance in the paper (e.g., `Fig1_xxx.py`, `Fig2_xxx.py`, etc.) for easy reference, execution, and verification.

Running these scripts will automatically generate, process, and output the experimental data required for the figures, resulting in **source data files for plotting** (such as `.csv` or `.xlsx` formats). The figures themselves are created using `matplotlib`, but this project **does not include the matplotlib plotting code for final figure rendering**. Users can reproduce the figures using their preferred plotting tools (recommended: `matplotlib`) based on the output data.

The paper involves several types of complex network models, including:

* **ER Network** (Erdős–Rényi random graph)
* **WS Network** (Watts–Strogatz small-world network)
* **BA Network** (Barabási–Albert scale-free network)

In the implementation, these network types differ only slightly in code, specifically in a single line that generates the network structure (e.g., `G = nx.erdos_renyi_graph(...)`). The rest of the experimental flow (simulation, statistics, output) remains identical.

To avoid redundancy, this project **only provides the full implementation for the He network (heterogeneous network)** as a representative. If readers want to reproduce results based on ER, WS, or BA networks, they only need to replace the network generation line in the scripts with the corresponding `networkx` function. Example replacements are as follows:

```python
# Replace with ER network
G = nx.erdos_renyi_graph(N, p)

# Replace with WS network
G = nx.watts_strogatz_graph(N, k, beta)

# Replace with BA network
G = nx.barabasi_albert_graph(N, m)
```

## Project Structure Example

Quiescent-Behaviors/

├── Fig3/

│   ├── Fig3\_He.py      # Evolutionary dynamics on heterogeneous network in Figure 3

│   └── ...             # Other subfigure scripts in Figure 3

├── Fig4/

│   └── ...             # Scripts for subfigures in Figure 4

├── Fig5/

│   └── ...             # Scripts for Figure 5

├── SI/

│   └── ...             # Code used in the supplementary information

└── README.md            # Project description

## Contact Information

If you have any questions during usage, please feel free to contact the author:

* Email: [cyluo2005@gmail.com](mailto:cyluo2005@gmail.com)
