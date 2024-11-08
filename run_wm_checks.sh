#! /bin/bash -e

START=0
END=10
CHECK_WM=true
GEN_IMG=true
OUTPUT_FOLDER=./results

function usage {
    echo "Usage: $(basename $0) [options] config [device]"
    echo "Options:"
    echo "    -b Base (no watermark)"
    echo "    -s [num] Start prompt number"
    echo "    -e [num] End prompt number"
    echo "    -n No image generation, just check watermarks"
    echo "    -o [path] output folder"
    echo "    -h help"
}

while getopts 'bs:e:o:nh' opt; do
  case "$opt" in
    b)
      CHECK_WM=false
      ;;

    s)
      START=$OPTARG
      ;;

    e)
      END=$OPTARG
      ;;

    n)
      GEN_IMG=false
      ;;
   
    o)
      OUTPUT_FOLDER=$OPTARG
      ;;

    ?|h)
      usage
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

CONFIG_FILE=config/$1.yaml

if [ $# -ge 2 ]; then
  DEVICE=$2
elif [ "$DEVICE" == "" ]; then
  DEVICE="cuda:0"
fi

if [ ! -r "$CONFIG_FILE" ]; then
    echo "Cannot read config file $CONFIG_FILE"
    exit 2
fi

if [ ! -d results ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

if $GEN_IMG; then
    echo Generating images for $1
    python generate_images.py $CONFIG_FILE device=$DEVICE start=$START end=$END
fi

if $CHECK_WM; then
    echo Checking watermarks for $1
    python check_watermark.py $CONFIG_FILE device=$DEVICE start=$START end=$END | tee "$OUTPUT_FOLDER/${1}.txt"
    cat "$OUTPUT_FOLDER/${1}.txt"
fi
