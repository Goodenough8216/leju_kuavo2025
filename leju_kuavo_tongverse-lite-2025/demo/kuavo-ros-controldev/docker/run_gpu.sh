#!/bin/bash
xhost +

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PARENT_DIR/.ccache"

DIR_HASH=$(echo "$PARENT_DIR" | md5sum | cut -c1-8)
echo "Directory $PARENT_DIR hash: $DIR_HASH"
CONTAINER_NAME="kuavo_container_gpu_${DIR_HASH}"
# 自动使用最新的镜像版本
IMAGE_NAME=$(docker images kuavo_opensource_mpc_wbc_img --format "{{.Repository}}:{{.Tag}}" | sort -V | tail -n1)

show_container_info() {
    local div_line="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "\n$div_line"
    echo -e "📌 \033[34mContainer Info\033[0m: $CONTAINER_NAME"
    echo -e "📂 \033[32mWorking Directory\033[0m:"
    echo -e "   $PARENT_DIR"
    echo -e "🔗 \033[33mMounted Volumes\033[0m:"
    # docker inspect -f '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' $CONTAINER_NAME

    docker inspect -f '{{range .Mounts}}   {{.Source}} → {{.Destination}}{{println}}{{end}}' $CONTAINER_NAME
    echo -e "$div_line\n"
}


if [[ $(docker ps -aq -f ancestor=${IMAGE_NAME} -f name=${CONTAINER_NAME}) ]]; then
    echo "Container '${CONTAINER_NAME}' based on image '${IMAGE_NAME}' is already exists."
    if [[ $(docker ps -aq -f status=exited -f name=${CONTAINER_NAME}) ]]; then
        echo "Restarting exited container '$CONTAINER_NAME' ..."
        docker start $CONTAINER_NAME
    fi
    show_container_info
    echo "Exec into container '$CONTAINER_NAME' ..."
    docker exec -it $CONTAINER_NAME zsh
else
	docker run -it --net host --gpus all \
        --runtime nvidia \
		--name $CONTAINER_NAME \
		--privileged \
		-v /dev:/dev \
		-v "${HOME}/.ros:/root/.ros" \
		-v "$PARENT_DIR/.ccache:/root/.ccache" \
		-v "$PARENT_DIR:/root/kuavo_ws" \
		-v "${HOME}/.config/lejuconfig:/root/.config/lejuconfig" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all,display \
        -e CARB_GRAPHICS_API=vulkan \
        -e GDK_SYNCHRONIZE=1 \
		--group-add=dialout \
		--ulimit rtprio=99 \
		--cap-add=sys_nice \
		-e DISPLAY=$DISPLAY \
		-e ROBOT_VERSION=40 \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		${IMAGE_NAME} \
		zsh
fi
