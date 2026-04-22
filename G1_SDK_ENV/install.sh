#!/bin/bash
set -e

ARG_USER=gdk # gdk or gma
ARG_USE_UV=false
ARG_NO_UPDATE_BASHRC=false

while (($#)); do
    case "$1" in
    --use_uv)
        ARG_USE_UV=true
        shift
        ;;
    --no_update_bashrc)
        ARG_NO_UPDATE_BASHRC=true
        shift
        ;;
    --user)
        ARG_USER=$2
        shift 2
        ;;
    *)
        echo "Usage: bash scripts/install.sh [--use_uv] [--update_bashrc] [--user gdk|gma]"
        exit 1
        ;;
    esac
done

if [ "$ARG_USE_UV" == "true" ]; then
    pip_cmd="uv pip"
    which uv
else
    pip_cmd="pip"
fi

# Function to detect the system architecture
function get_architecture() {
    local arch=$(uname -m)
    if [[ $arch == *"x86_64"* ]]; then
        echo "X86_64"
    elif [[ $arch == *"arm"* || $arch == *"aarch64"* ]]; then
        echo "AARCH64"
    else
        echo "Unknown"
    fi
}

# Main execution flow
ARCH=$(get_architecture)
if [[ $ARCH == "Unknown" ]]; then
    echo "Unable to identify the current system architecture"
    exit 1
fi

if [[ $ARG_USER == "gma" ]]; then
    agibot_home_dir=$HOME/.cache/agibot
    gdk_home_dir=$agibot_home_dir/gdk
else
    agibot_home_dir=$(cd "$(dirname "$0")" && pwd)
    gdk_home_dir=$agibot_home_dir/a2d_sdk
fi

# 安装sshpass
function install_sshpass() {
    # 检查是否已安装sshpass
    if ! command -v sshpass &> /dev/null; then
        sudo apt update
        sudo apt-get install sshpass -y
    else
        echo "sshpass 已安装"
    fi
}

# 将a2d上的package拷贝到本地并解压缩
function copy_sdk_package() {
    install_sshpass
        
    curl -o $agibot_home_dir/a2d_sdk_server.tar.gz http://10.42.0.101:8849/gdk_server_installer.tar.gz
    tar -zxf $agibot_home_dir/a2d_sdk_server.tar.gz -C $agibot_home_dir
    # for gdk user, keep the sdk package in the current script dir, maybe change in the future
    # for gma user, change the cache dir to $HOME/.cache/agibot/gdk
    
    mv $agibot_home_dir/gdk_server_installer $gdk_home_dir
    
    rm $agibot_home_dir/a2d_sdk_server.tar.gz
}

function install_sdk_package() {
    if [[ $ARCH == "X86_64" ]]; then
        $pip_cmd install $gdk_home_dir/cosine_bus-2.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall
    elif [[ $ARCH == "AARCH64" ]]; then
        $pip_cmd install $gdk_home_dir/cosine_bus-2.0.0-cp310-cp310-linux_aarch64.whl --force-reinstall
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi

    # 安装genie_msgs_pb
    $pip_cmd install $gdk_home_dir/genie_msgs_pb-0.8.0-py3-none-any.whl --force-reinstall --no-deps

    # 安装a2d_sdk
    a2d_sdk_whl=$(ls -t $gdk_home_dir/a2d_sdk-*.whl | head -n 1)
    $pip_cmd install $a2d_sdk_whl --force-reinstall

    if [[ $ARCH == "X86_64" ]]; then
        # 安装forward_package
        mkdir -p $gdk_home_dir/forwarder
        tar -zxf $gdk_home_dir/forwarder_x86_v1.7.0.tar.gz -C $gdk_home_dir/forwarder
    fi
    
    rm $a2d_sdk_whl
    rm $gdk_home_dir/genie_msgs_pb-0.8.0-py3-none-any.whl
    rm $gdk_home_dir/cosine_bus-2.0.0-cp310-cp310-linux_x86_64.whl
    rm $gdk_home_dir/cosine_bus-2.0.0-cp310-cp310-linux_aarch64.whl
    rm $gdk_home_dir/forwarder_x86_v1.7.0.tar.gz
}

function modify_bashrc() {
    if [ "$ARG_NO_UPDATE_BASHRC" == "false" ]; then
        read -p "是否要修改 bashrc 来source env.sh? (y/n): " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            # 检查是否存在 .bashrc 文件
        if [ -f "$HOME/.bashrc" ]; then
            # 备份原文件
            if [ ! -f "$HOME/.bashrc.backup" ]; then
                cp "$HOME/.bashrc" "$HOME/.bashrc.backup"
            fi
            
            # 检查是否存在环境变量
            if ! grep -q "source $gdk_home_dir/env.sh" "$HOME/.bashrc"; then
                echo "source $gdk_home_dir/env.sh" >> "$HOME/.bashrc"
            fi
            
            echo "bashrc 已修改,原文件已备份为 .bashrc.backup"
        else
            echo "未找到 .bashrc 文件"
            fi
        fi
    fi
}

function check_old_msgs() {
    CHECK_DIR=/usr/local/genie/
    if [ -d "$CHECK_DIR" ]; then
        echo "检查到历史版本遗留文件，需要删除"
        rm -rf $CHECK_DIR
    fi
}

function install_a2d_sdk_server() {
    if [ -d "$gdk_home_dir" ]; then
        rm -rf $gdk_home_dir
    fi
    mkdir -p $agibot_home_dir
    copy_sdk_package
    install_sdk_package
    modify_bashrc
    check_old_msgs
}

install_a2d_sdk_server
echo "安装完成"
