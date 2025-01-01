while getopts g:n:p: flag
do
	case "${flag}" in
		g) gpu=${OPTARG};;
		n) number=${OPTARG};;
		p) port=${OPTARG};;
	esac
done
echo "Running container ser$number on gpus $gpu visible at port $port";

docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=$gpu --runtime=nvidia --userns=host --shm-size 64gb -v /work/lucas.ueda/:/workspace/lucas.ueda/ -p $port --name ser$number interspeech_ser:latest /bin/bash
