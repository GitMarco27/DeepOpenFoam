#!/bin/sh

# Quit script if any step has error:
set -e

# Cleaning the working path
python3 scripts/clean.py

# Changing Boundary Conditions
python3 scripts/change_bd.py --a 0 --U 45

# Changing geometry
python3 scripts/generator.py

# Generate the mesh from script (msh2 format is currently the latest gmshToFoam recognizes):
gmsh -3 -o main.msh -format msh2 mesh/bl_geometry.geo -nopopup > log.simulation

# Convert the mesh to OpenFOAM format:
gmshToFoam  main.msh -case case > log.simulation

# Adjust polyMesh/boundary:
changeDictionary -case case > log.simulation

# MPI
decomposePar -case case > log.simulation

# Finally, run the simulation:
mpirun -np 10 simpleFoam -case case -parallel > log.simulation

# Reverse the mesh splitting
reconstructPar -case case > log.simulation

# Saving results
python3 scripts/postprocessing.py
