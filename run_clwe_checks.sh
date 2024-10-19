#! /bin/bash -e

START=0
END=10

CONFIG_FILE=config/$1.yaml

if [ ! -r "$CONFIG_FILE" ]; then
    echo "Usage $0 [conf name]"
    exit 1
fi

if [ ! -d results ]; then
    mkdir results
fi

if [ "$2" != "nogen" ]; then
    echo Generating images for $1
    python generate_images.py $CONFIG_FILE device=$DEVICE start=$START end=$END
fi

if [ "$1" != "base" ]; then
    echo Checking watermarks for $1
    python check_watermark.py $CONFIG_FILE device=$DEVICE start=$START end=$END no_wm_path=images/no_wm | tee results/${1}.txt
    cat results/${1}.txt
fi
