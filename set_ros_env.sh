# 1. Source ROS2 Jazzy 官方路径
    source /opt/ros/jazzy/setup.bash

    # 2. 设置 Isaac Sim 路径 (根据你的安装位置调整，通常在 .local 下)
    export ISAAC_SIM_PATH=$HOME/.local/share/ov/pkg/isaac-sim-4.2.0  # 请确认你的版本号

    # 3. 关键：将 ROS2 的库路径加入到 LD_LIBRARY_PATH，这样 Isaac 内置的 Python 3.10 才能调用 ROS2 动态库
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/jazzy/lib
    ```
