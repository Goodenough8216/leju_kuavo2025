# #!/bin/bash

commands=("roscore" 
          "rosrun dynamic_biped highlyDynamicRobot_node"
          "python3 $CATKIN_WS_PATH/src/noitom_hi5_hand_udp_python/scripts/noitom_hand_publish.py "
          "cd $CATKIN_WS_PATH/src/handcontrollerdemorosnode && python3 src/main.py --debug"
          )
window_names=("roscore" 
          "highlyDynamicRobot_node" 
          "noitom_hand_publish" 
          "handcontrollerdemorosnode"
          )

if ! command -v tmux &> /dev/null
then
    echo "tmux could not be found, please install it first."
    exit
fi

session_name="arm_control_session"
tmux new-session -d -s $session_name

for i in "${!commands[@]}"; do
  window_name="${window_names[$i]}"
  if [ $i -eq 0 ]; then
    continue
  else
    tmux new-window -t $session_name -n $window_name
    if [ $window_name == "highlyDynamicRobot_node" ]; then
      tmux send-keys -t $session_name:$window_name "sudo su" C-m
    fi
    tmux send-keys -t $session_name:$window_name "source $CATKIN_WS_PATH/devel/setup.bash" C-m
    tmux send-keys -t $session_name:$window_name "${commands[$i]}" C-m
  fi
  sleep 0.5
done

tmux attach-session -t $session_name
