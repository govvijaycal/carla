#docker run -it -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.9

docker run -it -p 2000-2002:2000-2002 --gpus all carlasim/carla:0.9.9 /bin/bash CarlaUE4.sh -benchmark -fps=10
