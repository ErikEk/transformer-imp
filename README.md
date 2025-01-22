# transformer-imp

# Dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

sudo docker run --name gs_container5 -p 8888:8888 --gpus all -it -v $(pwd):/app getting-started
USE -d to run in background

# Build the image
sudo docker build -t getting-started .

# connect with:
sudo docker exec -it gs_container /bin/bash

# List containers
docker ps -a

# start stop
sudo docker start gs_container
sudo docker stop gs_container

# Remove
sudo docker rm gs_container

# Guide
https://www.youtube.com/watch?v=0H2miBK_gAk&ab_channel=PatrickLoeber