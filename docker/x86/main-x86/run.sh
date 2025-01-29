# export DATA_PATH=$(pwd)/bop_datasets
cd main
sudo docker run --gpus all --env-file ./env.list -v /$(pwd)/bop_datasets/:/home/bop_datasets/ main
