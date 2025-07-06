SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$(realpath "$SCRIPT_DIR/")
mkdir -p "$PROJECT_DIR/.ccache"

# 生成目录哈希值用于容器命名
DIR_HASH=$(echo "$PROJECT_DIR" | md5sum | cut -c1-8)
CONTAINER_NAME="kuavo_isaac_${DIR_HASH}"
echo "Directory $PROJECT_DIR hash: $DIR_HASH"

xhost +
docker run --rm --name $CONTAINER_NAME -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --network=host \
    --ulimit rtprio=99 \
    --privileged \
    -e "PRIVACY_CONSENT=Y" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all,display \
    -e CARB_GRAPHICS_API=vulkan \
    -e GDK_SYNCHRONIZE=1 \
    -e ROBOT_VERSION=45 \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -e DISPLAY \
    --group-add=dialout \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$PROJECT_DIR/.ccache:/root/.ccache" \
    -v $PROJECT_DIR:/TongVerse/biped_challenge:rw \
    kuavo_tv bash