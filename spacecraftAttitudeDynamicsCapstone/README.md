# Spacecraft Attitude Dynamics and Control Capstone Project
Code for the capstone project of ASEN 5010 - Spacecraft Attitude Dynamics and Control at
CU Boulder. The project focused on developing attitude guidance and control algorithms for a small 
spacecraft orbiting Mars. A detailed description of the project and the mathematical 
background for each algorithm is provided in [Documentation](https://github.com/Jmiceli17/Prototypes/tree/main/spacecraftAttitudeDynamicsCapstone/Documentation).

# Set Up
Clone this repository\
`git clone https://github.com/Jmiceli17/Prototypes.git`

Navigate to this directory\
`cd spacecraftAttitudeDynamicsCaptsone`

Create and initialize a conda enviornment using the provided `environment.yml` This will create a conda environment and install the python dependencies needed to run this code.\
`conda env create -f environment.yml`

#### NOTE
If running in WSL2, you will have to install and start [XLaunch/VcXsrv](https://sourceforge.net/projects/vcxsrv/) (or some other X-server) in order for the plots to display.

# Usage
Activate the conda environment\
`conda activate adcs-capstone`

The main mission simulation is defined in `MissionSimulation.py` Running this file runs the entire simulation and
produces plots of the results.\
`python MissionSimulation.py`

Other simulations can be found in `PointingSimulations.py`\
`python PointingSimulations.py`
