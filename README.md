# 🌕 Luna Rover 🌕

**Project:** This project is an extension of the [Luna Rover](https://github.com/wmcgloin/luna_rover) project, which was developed by Shriya Chinthak, Jorge Bris Moreno, Billy McGloin, Eric Dapkus, Kangheng Liu.

**Authors:** Jorge Bris Moreno, Billy McGloin

**Supervisors:** Dr. J. Hickman, Dr. Prabal Saxena.

This project aims to develop a simulation of a lunar rover exploring the lunar surface, with the objective of finding and gathering resources. The rover will have to navigate the lunar surface, avoiding obstacles, and managing its energy consumption. The goal is to develop a simulation space that recreates the moon surface with different types of resources, and train different RL algorithms to see if they would be able to control the rover to not only find the resources, but interact with the environment while controlling its battery and safety.

**Note:** In this iteration of the project, some assumptions have been made to simplify the problem. These assumptions are detailed in our [report](report.html).

## Project structure

The full project is structured as detailed as follows:

(🛠️) = Files are essential to the overall build of the project, but not the content

(🚫) = Files were created throughout the project, but no longer necessary for running the simulation

(📝) = The explanation for these files can be found in our [report](report.html)

```
.
├── LICENSE (🛠️)
├── MANIFEST.in (🛠️)
├── README.md (You are here 👋)
├── archive (🚫)                  --> files in this folder have been archived. No longer in use
│   ├── dqn.py
│   ├── dqn_output
│   │   ├── dqn_gameplay.gif
│   │   ├── dqn_policy_visualization.png_day0.png
│   │   ├── dqn_policy_visualization.png_day15.png
│   │   ├── dqn_policy_visualization.png_day22.png
│   │   ├── dqn_policy_visualization.png_day7.png
│   │   └── dqn_training_progress.png
│   ├── dqn_train.py
│   ├── example_usage.py
│   ├── rewardStructure.md
│   └── solvers.py
├── bibliography.bib              --> BibTex citations for the report
├── docs (🚫) 
│   └── Luna_WriteUp.qmd                     
├── images 
│   └── env_plot.png              --> Environment image located in the report
├── policy_gifs                   --> .gifs of RL simulations for each tested solver
│   ├── algorithm_policy.gif
│   ├── appo_policy1.gif
│   ├── dqn_policy1.gif
│   ├── ppo_policy1.gif
│   ├── sac_policy1.gif
│   └── sac_policy2.gif
├── report.html                   --> Rendered final report. To view, simply open in your local browser
├── report.qmd                    --> Code file of the final report. The rendered output is the html version
├── setup.py (🛠️)
├── solvers.ipynb                 --> This file runs all the solvers and saves results to policy_gifs
└── src
    └── lunabot
        ├── __init__.py (🛠️)
        ├── geosearch.py (📝)
        ├── robot.png             --> Image of our rover in the pygame visualization of the simulation
        └── utils.py (📝)
```

## Usage

After cloning the repository, navigate to the root directory and run the following command to install the package:

```{bash}
pip install -e .
```

You can then run `solvers.ipynb` file, which will train assorted solvers and the visualize in realtime, saving them has .gif files. 

## Findings 

For an extensive explanation of the project, specifically the environment, agent and rewards, solvers, and results, please read through our [report](report.html). 