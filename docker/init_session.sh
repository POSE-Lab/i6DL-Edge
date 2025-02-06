#!/usr/bin/env bash

#get name of session from command line 
# if the session is not already running, create a new session,
# otherwise just attach to it.
if [ $# -lt 2 ]; then
  echo -e "Please provide 2 arguments:\nsession name, image tag"
  exit 1
fi

ARCH=$(lscpu | sed -n '1p' | awk '{print $2}')
if [ "$ARCH" == "aarch64" ]; then
  IMAGE="main-arm"
elif [ "$ARCH" == "x86_64" ]; then
  IMAGE="main-x86"
fi

TAG=$2
#CAM_RES=$3
ENV_FILE=$4

tmux has-session -t $1
if [ $? -ne 0 ]; then
  echo "Running on $ARCH architecture, deploying $IMAGE:$TAG"
  echo "Creating session $1"
  #tmux select-layout tiled
  tmux new-session -d -s $1
  tmux split-window -h
  tmux split-window -v
  tmux select-pane -t 0
  tmux split-window -v

  tmux send-keys -t 1 "./run_container.sh $IMAGE $TAG $ENV_FILE" Enter
  tmux select-pane -t 1
  if [ "$ARCH" == "aarch64" ]; then
    tmux send-keys -t 1 "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1" Enter 
  fi
  sleep 3
  tmux send-keys -t 2 './run_new.sh' Enter
  tmux send-keys -t 0 "./run_new.sh" Enter
  sleep 5
  tmux send-keys -t 1 "./entrypoint.sh" Enter

  tmux set -g mouse on
  tmux -2 attach-session -d
  
else 
  echo "Attaching to session $1"
  sleep 1
  tmux attach -t $1
fi