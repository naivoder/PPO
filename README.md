# Proximal Policy Optimization (Continuous)

## Overview

🚧 🛠️👷‍♀️ 🛑 Under construction...

## Setup

### Required Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'LunarLanderContinuous-v2'
```

Notes: Reward scaling appears to work really well for some environments (BipedalWalker) but it might be limiting the upper bound of performance on some other environments. I've increased the number of episodes to 50k for the Mujoco environments, if that gives the agent enough time to learn I'll rerun on the Gymnasium ones. Examples in the paper train for *millions* of timesteps...

<table>
    <tr>
        <td>
            <p><b>Pendulum-v1</b></p>
            <img src="environments/Pendulum-v1.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MountainCarContinuous-v0</b></p>
            <img src="environments/MountainCarContinuous-v0.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>LunarLanderContinuous-v2</b></p>
            <img src="environments/LunarLanderContinuous-v2.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Pendulum-v1_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MountainCarContinuous-v0_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/LunarLanderContinuous-v2_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Pusher-v4</b></p>
            <img src="environments/Pusher-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Reacher-v4</b></p>
            <img src="environments/Reacher-v4.gif" width="250" height="250"/>
        </td>
       <td>
            <p><b>InvertedPendulum-v4</b></p>
            <img src="environments/InvertedPendulum-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Pusher-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Reacher-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/InvertedPendulum-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BipedalWalker-v3</b></p>
            <img src="environments/BipedalWalker-v3.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>InvertedDoublePendulum-v4</b></p>
            <img src="environments/InvertedDoublePendulum-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Walker2d-v4</b></p>
            <img src="environments/Walker2d-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BipedalWalker-v3_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/InvertedDoublePendulum-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Walker2d-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Ant-v4</b></p>
            <img src="environments/Ant-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>HalfCheetah-v4</b></p>
            <img src="environments/HalfCheetah-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Swimmer-v3</b></p>
            <img src="environments/Swimmer-v3.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Ant-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HalfCheetah-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Swimmer-v3_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
