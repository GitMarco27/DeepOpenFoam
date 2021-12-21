# Install gmsh, foam and dependencies
apt-get update

# Python3
apt-get install python3 -y
apt-get install pip -y

# Foam
curl -s https://dl.openfoam.com/add-debian-repo.sh | bash
apt-get install openfoam2106-default -y

# gmsh
pip3 install gmsh

# Python packages
pip3 install -r requirements.txt