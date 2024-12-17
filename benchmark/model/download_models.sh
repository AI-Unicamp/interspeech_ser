#!/bin/bash

# Check if exactly one or more arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {categorical|attributes|all}"
    exit 1
fi

# Function definitions for each task
do_categorical() {
    echo "Downloading categorical model"
    wget https://lab-msp.com/MSP-Podcast_Competition/IS2025/models/cat_ser.zip 
    unzip cat_ser.zip
    rm cat_ser.zip
}

do_attributes() {
    echo "Downloading attributes model"
    wget https://lab-msp.com/MSP-Podcast_Competition/IS2025/models/dim_ser.zip 
    unzip dim_ser.zip
    rm dim_ser.zip
}

# Main logic to process the input argument/s
for arg in "$@"
do
    case $1 in
        categorical)
            do_categorical
            ;;
        attributes)
            do_attributes
            ;;
        all)
            do_categorical
            do_attributes
            ;;
        *)
            echo "Invalid argument: $1"
            echo "Usage: $0 {categorical|attributes|all}"
            exit 2
            ;;
    esac
done

exit 0
