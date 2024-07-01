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
        <td>
            <p><b>BipedalWalker-v3</b></p>
            <img src="environments/BipedalWalker-v3.gif" width="250" height="250"/>
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
        <td>
            <img src="metrics/BipedalWalker-v3_running_avg.png" width="250" height="250"/>
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
       <!--  <td>
            <p><b>Asteroids-v5</b></p>
            <img src="environments/Asteroids-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Breakout-v5</b></p>
            <img src="environments/Breakout-v5.gif" width="250" height="250"/>
        </td>
    </tr> -->
    <tr>
        <td>
            <img src="metrics/Pusher-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Reacher-v4_running_avg.png" width="250" height="250"/>
        </td>
        <!-- <td>
            <img src="metrics/Asteroids-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Breakout-v5_running_avg.png" width="250" height="250"/>
        </td> -->
    </tr>
</table>
<!-- <table>
    <tr>
        <td>
            <p><b>BeamRider-v5</b></p>
            <img src="environments/BeamRider-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Centipede-v5</b></p>
            <img src="environments/Centipede-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>DonkeyKong-v5</b></p>
            <img src="environments/DonkeyKong-v5.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BeamRider-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Centipede-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/DonkeyKong-v5_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Frogger-v5</b></p>
            <img src="environments/Frogger-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>KungFuMaster-v5</b></p>
            <img src="environments/KungFuMaster-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MarioBros-v5</b></p>
            <img src="environments/MarioBros-v5.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/InvertedDoublePendulum-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KungFuMaster-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MarioBros-v5_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>MsPacman-v5</b></p>
            <img src="environments/MsPacman-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>SpaceInvaders-v5</b></p>
            <img src="environments/SpaceInvaders-v5.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tetris-v5</b></p>
            <img src="environments/Tetris-v5.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/MsPacman-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/SpaceInvaders-v5_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Tetris-v5_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table> -->

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
