#!/bin/bash

if [ -z "$1" ] ; then
    echo "Usage:  $0    <image name>    [optional args for docker image]"
    echo "Missing Docker image id.  exiting"
    exit 1
fi

#image_id="$1"
#shift
#cmd="$@"

working_dir=$(pwd -P)


docker run -it   \
    -v "$working_dir:/home/ubuntu/dev" \
    -v ~/datasets:/data \
    -p 80:80 \
    -p 2222:22 \
    -p 4040-4060:4040-4060 \
    -p 5901:5901 \
    -p 6006:6006 \
    -p 8000:8000  \
    -p 8001:8001  \
    -p 8002:8002  \
    -p 8008:8008  \
    -p 8080:8080  \
    -p 8081:8081  \
    -p 8787:8787  \
    -p 8888:8888  \
    -p 9000:9000  \
    "$@"
