# Operative Dimensions in Unconstrained Connectivity of Recurrent Neural Networks
This is the official implementation of **Operative dimensions in unconstrained connectivity of recurrent neural networks** (NeurIPS 2022).

(Renate Krause, Matthew Cook, Sepp Kollmorgen, Valerio Mante* and Giacomo Indiveri*, * Equal contribution)


The code (provided in matlab (code_matlab/..) and python (code_python/..)) shows you how to:
* (script s1) reproduce our results on high-variance dimensions (Figure 1c-e, h-j).
* (script s2) generate local operative dimensions
* (script s3) reproduce our results on global operative dimensions (Figure 3b-d, e-g).
* (script s4) compare network output and trajectories of full-rank to reduced-rank network (not shown in paper).

Additionally, we provide:
* 20 pretrained models for context-dependent integration (1) and sine wave generation each (2) (see data/pretrained_networks/).
* local operative dimensions to generate the global operative dimensions for the first of the pretrained networks for each task (see data/local_operative_dimensions/).

## Requirements
* code tested and developed using matlab R2019b/python3.8

## Usage
* Note I: run scripts from within the folder "code_matlab"/"code_python"
* Note II: In each script (s1-4) you might have to adjust the paths at the top of the script to allow your system to find your code and our provided data (data located in folder operativeDimensions/data/...).
* Note III: script 2 (s2) allows you to calculate the local operative dimensions. The local operative COLUMN dimensions can be calculated explicitly and it only takes a few seconds. However, the local operative ROW dimensions require a numerical optimization procedure and therefore take longer to generate (see section "Required computational resources" in manuscript for more information on required time).
* Note IV: If you run script 2 (s2), you generate a new set of local operative dimensions. The name of the output file contains the current date and time to ensure you do not accidentally overwrite an existing file and is printed at the end of the script. In case you want to use the newly generated local operative dimensions rather than the provided ones, you have to modify the variable "inputfilename" in script 3 and 4 (s3/4, in 3rd cell from top) to lead the scripts to the new file.

To reproduce the main results of our paper, run the following scripts in matlab/python.
##### Results on high-variance dimensions:
s1_analysis_high_variance_dimensions.m/.ipynb

##### Generate local operative dimensions:
s2_generate_operative_dimensions.m/.ipynb

##### Results on global operative dimensions:
s3_analysis_global_operative_dimensions.m/.ipynb

##### Run full- and reduced-rank network to compare output and population activities:	
s4_run_reduced_rank_network.m/.ipynb


## Results
### Global operative dimensions identify a low-dimensional subspace in W that is sufficient for the RNN to achieve the original performance
![operative_dimensions](/figures_opDims.png "Operative dimensions")

**(a)** Definition of operative dimensions based on local recurrent dynamics along a condition average trajectory. Arrows show the recurrent contribution to the dynamics for the full-rank and several reduced-rank networks (colors, see legend). Local operative dimensions maximize $\Delta f$. Here inputs $\textbf{u}_t$ and noise $\boldsymbol{\sigma}_t$ are set to zero. **(b)** Rank of global operative column dimensions, estimated with PC analysis on concatenated local operative dimensions. **(c)** Network output cost of networks with reduced-rank weight matrix $\textbf{W}_k^{OP}$ for $k=1:N$. **(d)** State distance between trajectories in the full-rank network and in networks with reduced-rank weight matrix $\textbf{W}_k^{OP}$. **(b)**-**(d)** Based on global operative column dimensions and averaged over 20 networks per task; shaded area: $mad$. Network output cost obtained with internal and input noise, state distance without any noise. **(e)**-**(g)** Same as **(b)**-**(d)** for global operative row dimensions.

## License
Code on operative dimensions is licensed under the MIT license, see the LICENSE_MIT file.

## References
Code to run RNN (rnnFramework) is adapted from (1).

(1) Valerio Mante, David Sussillo, Krishna V Shenoy, and William T Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. nature, 503(7474):78–84, 2013.

(2) David Sussillo and Omri Barak. Opening the black box: low-dimensional dynamics in high dimensional recurrent neural networks. Neural computation, 25(3):626–649, 2013.
