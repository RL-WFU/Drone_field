# Drone_field
*Final Drone Repository Ashley Peake and Joe McCalmon WFU 2020*

This repository contains everything neccessary to run our drone navigation algorithm based on two deep learning networks.

## Running

To run the code using the provided network weights, simply run `run.py`. If neccessary, you can test drone navigation using different weights by changing the 
arguments --search_weights and --trace_weights accordingly. Our trained weights can be found in the `Weights` folder

Basic episode configuration values are set in `configurationSimple.py`, including the envionment image that the simulation is run on. This file needs to be
updated based on your testing needs.

## The Episode

Running `run.py` calls `Testing/full_testing.py`. This file contains a function for excecuting a complete episode. In an episode, the drone navigates an area by 
repeating a sequence of subtasks: target selection, search, and trace. The implementation of these subtasks, along with other functions relating to testing, 
can be found in `Testing/testing_helper.py`. Details of the episode and subtasks can be found in comments in these files.

## The Environment 

Episodes are run according to the environment for each subtask. At its most fundamental level, the environment provides the state and reward for the agent 
at each timestep based on the action it takes. The environments for the subtasks can be found in the `Environment` folder. Each subtask environment is a subclass of 
the base Env class found in `base_env.py`. The specifics of these environments are included as comments in their respective files. 

For SLAM (Simultaneous Localization and Mapping), we keep track of the path taken by the drone and save the information it collects at each timestep as a map. Its 
path is recorded in a Visited object, defined in `Environment/visited.py`. Similarly, the cumulative information seen by the drone is stored as a Map object, 
defined in `Environment/map.py`. Greater details of these classes are explained as comments in their files. 

## Updating for Field Implementation

The simulated drone navigates its environment according to array indices. Updating this to gps coordinates will be the most substantial change for field use.
The drone also currently gets classsification information based on RGB classification of an image. This happens in `ICRSsimulator.py`, but will need to be
altered to instead be based on images taken by the drone. Comments throughout the code further note obvious changes that should be made, but there are likely more 
than what we've indicated. Primarily pay attention to the drone position and state information, and updates shouldn't be that complicated.
