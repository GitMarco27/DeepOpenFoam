#!/bin/sh

# Quit script if any step has error:
set -e

# Cleaning the working path
python3 clean.py

python3 change_bd.py --a 18 --U 45

# Generate the mesh from script (msh2 format is currently the latest gmshToFoam recognizes):
gmsh -3 -o main.msh -format msh2 mesh/bl_geometry.geo
# Convert the mesh to OpenFOAM format:
gmshToFoam  main.msh -case case
# Adjust polyMesh/boundary:
changeDictionary -case case

# MPI
decomposePar -case case

# Finally, run the simulation:
mpirun -np 10 simpleFoam -case case -parallel

# Reverse the mesh splitting
reconstructPar -case case
