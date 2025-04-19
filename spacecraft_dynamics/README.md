# Spacecraft Dynamics
A Python package developed from various assignments and projects completed for courses in the Aerospace Engineering Department at CU Boulder.

## Features
- Modeling of satellite attitude dynamics
- Support for various kinds of actuators
    - Reaction wheels
    - Variable-Speed Control Moment Gyroscopes (VSCMGs)
    - None (control authority is assumed)
- Support for different control algorithms
- Support for different guidance algorithms


## Coming Soon
- Guidance algorithms as the primary interface to the satellite (this is how most 
3rd party software uses an ADCS!)
- Orbit propagation
    - Some of the example scenarios fake this by converting propagating spherical coordinates over 
    time and converting them to cartesian (no equations of motion are actually modeled though)

## Installation
This package is not on PyPI but you can install it directly from the repository:

```bash
# Clone the repository
git clone https://github.com/Jmiceli17/AttitudePrototypes.git
cd AttitudePrototypes/spacecraft_dynamics

# Option 1: Install directly
pip install .

# Option 2: Install in development mode (if you want to modify the code)
pip install -e .
```


## Quick Start


## Documentation


## Examples
See the `examples/` directory for usage examples:
- Attitude control simulations
- Actuator configurations
- Orbital scenarios
```bash
# Example usage
cd AttitudePrototypes/spacecraft_dynamics/examples
python MissionSimulation.py
```
#### NOTE
If running in WSL2, you will have to install and start [XLaunch/VcXsrv](https://sourceforge.net/projects/vcxsrv/) (or some other X-server) in order for the plots to display.


## Development
For development, clone the repository and create a conda environment:

```bash
git clone https://github.com/Jmiceli17/AttitudePrototypes.git
cd AttitudePrototypes/spacecraft_dynamics
conda env create -f environment.yml
conda activate sc-dyn
pip install -e .
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
