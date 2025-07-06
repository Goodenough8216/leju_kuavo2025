
# don't put duplicate lines or lines starting with space in the history.
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=30000
HISTFILESIZE=300000000
# share history
#export PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize


# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

alias killa="sudo killall roslaunch"
alias pkilla="sudo pkill -f ros"
bind '"\e[A": history-search-backward' 
bind '"\e[B": history-search-forward'
export GIT_SSL_NO_VERIFY=1
alias exp="export DISPLAY=:0.0"
alias exp1="export DISPLAY=:1.0"
alias checkhw="sudo python3 /home/lab/kuavo/tools/check_tool/Hardware_tool.py"

