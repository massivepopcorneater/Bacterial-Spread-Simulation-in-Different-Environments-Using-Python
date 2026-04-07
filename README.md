# Simulation of Bacterial Spread in Different Environments using Python
## Introduction
A program that simulates the spread of bacteria in petri dish using Python. In this simulation, we account for temperature, pH and food variables. This program is made to be ran on home computers. Consequently, lower complexity methods will be used which sacrifices some realism. This project uses the Mesa Library an agent-based modelling library that allows for grid-based simulations, and Matplotlib, a library that allows users to create customized graphs. The application is divided into a file for the spread model, start program and agent/individual bacterium. This program aims to incorporate computer science, mathematics and biology concepts learned in CEGEP as well as research from peer-reviewed sources. 

## How to use
To run the program simply execute the run.py file. Once the figure is loaded, the simulation with instantly start at the optimal temperature for E.coli. The temperatures and pH can be adjusted using the sliders on the bottom left of the figure. On the bottom right, there are buttons to pause and reset the simulation. On the multigrid, green tiles are bacteria while the red dots are nutrients. The deeper the green, the more the bacteria on the tile and vice-versa.

## Computer Science Concepts
### Agent-Based Modelling
This simulation uses agent-based modelling, which is a simulation method that allows for behavior to be defined on an individual bacteria scale instead of a more global one. It uses agents, which in our case is E.coli bacteria, where each individual agent has its own unique behavior.

## Biology Concepts
### Protein Denaturation 
This program utilizes the concept of protein denaturation. Protein denaturation occurs when the temperature of the cell is high enough to make protein unfold itself which causes the protein to lose its function. In turn, without essential protein functionalities, the cell dies. In the application, the cell instantly dies once the cell reaches a temperature threshold.

### Lag Phase
Each cell in the simulation has a lag phase. This concept prevents cells from dividing until a certain period of time has passed. While in the lag phase, cells prepare essential elements for division such as synthesis of proteins, repair of damaged parts and adjusting to environmental conditions. In an ideal environment for the bacteria, the lag phase is shorter and longer in more challenging environments. The program implements this by adding a variable that slowly grows with time that increases the bacteria growth rate.

## Mathematics Concepts
### Ratkowski Model
The Ratkowski model 
### Gaussian Model

### Monod Equation

### Multiplicative Gamma Factor

## Pseudo Code

## Adding upon the program
- Add a realistic time scale
- Add more bacterias and environments
- Make the simulation more interactive by adding more UI elements
