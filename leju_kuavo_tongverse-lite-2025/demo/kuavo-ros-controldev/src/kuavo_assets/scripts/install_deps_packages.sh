#!/bin/bash

SCRIPT_DIR=$(dirname $(readlink -f "$0"))
PACKAGE_FILE=$SCRIPT_DIR/packages_with_versions.txt

ECHO_WARN() {
    local msg="$1"
    echo -e "\033[33m[Depend] ${msg}\033[0m" >&2
}

ECHO_ERR() {
    local msg="$1"
    echo -e "\033[31m[Depend] ${msg}\033[0m" >&2
}

# Check Depend packages.
# return 0 - All Packages are installed.
# return 1 - Not all packages are installed.
function check_packages_installed() {
     while IFS= read -r pkg || [[ -n "$pkg" ]]; do
        pkg_name=$(echo "$pkg" | cut -d'=' -f1 | cut -d':' -f1)
        pkg_version=$(echo "$pkg" | cut -d'=' -f2)
        installed_version=$(dpkg-query -W -f='${Version}\n' "$pkg_name" 2>/dev/null)
        if [[ "$installed_version" != "$pkg_version" ]]; then
            return 1
        fi
    done < <(cat "$PACKAGE_FILE")

    return 0
}

function install_packages() {
    sudo apt-get update || echo -e "\033[31mapt-get update failed, continuing...\033[0m" >&2
    while IFS= read -r pkg || [[ -n "$pkg" ]]; do
        pkg_name=$(echo "$pkg" | cut -d'=' -f1 | cut -d':' -f1)
        pkg_version=$(echo "$pkg" | cut -d'=' -f2)
        installed_version=$(dpkg-query -W -f='${Version}\n' "$pkg_name" 2>/dev/null)
        if [[ "$installed_version" != "$pkg_version" ]]; then
            ECHO_WARN "Installing depend package: [$pkg]"
            sudo apt-get install -y "$pkg"
            if [ $? -ne 0 ]; then
                ECHO_ERR "Install depend package [$pkg] failed."
                exit 1
            fi
        fi
    done < <(cat $PACKAGE_FILE; echo)
}

### Start
ECHO_WARN "Check and Install depend packages ..."
check_packages_installed "bash"
result=$?

if [ $result -eq 0 ]; then
    ECHO_WARN "All Packages are installed."
    exit 0
else
    install_packages
fi
exit 0