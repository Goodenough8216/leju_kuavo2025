sudo apt install libglfw3-dev

if [ -d "mujoco" ]; then
  echo "mujoco exists, skipping clone"
else
  git clone https://github.com/google-deepmind/mujoco.git
fi

cd mujoco
mkdir -p build && cd build
cmake ../
cmake --build .
sudo cmake --install .
