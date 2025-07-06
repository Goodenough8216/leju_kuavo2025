#!/bin/bash

# install ccache
sudo apt-get update
sudo apt-get install ccache -y

# configure ccache
add_ccache_to_rc_file() {
  local rc_file=$1  
    if [ -f "$rc_file" ]; then
        if ! grep -Fxq "export CC=\"ccache gcc\"" $rc_file ; then
            echo "export CC=\"ccache gcc\"" >> $rc_file
        fi

        if ! grep -Fxq "export CXX=\"ccache g++\"" $rc_file ; then
            echo "export CXX=\"ccache g++\"" >> $rc_file
        fi
    else
        echo "$rc_file not found"
    fi
}

add_ccache_to_rc_file ~/.bashrc
add_ccache_to_rc_file ~/.zshrc

echo "ccache enabled"

ccache --max-size=10G
