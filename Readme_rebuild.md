# Environment Setup
## Setup
This document is assuming that you have installed UnityHub or Unity application and have installed ml-agent library.
<br>Now, This tutorial will teach you how to build the SoccerTwo environment  which will directly interact with python api.
## Overview
There have four part to build the enviroment. 
* Setting the Academy.cs for soccer game
* build the binary file on environment. 
* gym wrapper Unity environment
* Test your environment for python

### 1. Setting the Academy
Open Editor and go to your file path (ml-agents/Example/SoccerTwos) and open the scenes. <br> 
click the Academy on left side (Fig 1)<br><br>
<img src="Academy.png" style="width:100%;" >

Find  the **Learning Brain** on right side and click the checkbox **control**.(Fig 2)<br>
<img src="control.png">

This process allow you to control the agent by the Learning Brain; Also, This setup allow you to train two brain in the training not just single brain but you need to custom your enviroment.

### 2. Binary File
After setup, you need to build the binary file; you can just use the binary file path on your python script; It will popup the window and open the scene for visualization.

1. Open Player Settings (menu: **Edit** > **Project Settings** > **Player**).
2. Under **Resolution and Presentation**:
   * Ensure that **Run in Background** is Checked.
   * Ensure that **Display Resolution Dialog** is set to Disabled.
3. Open the Build Settings window (menu:**File** > **Build Settings**).
4. Choose your target platform.
   * (optional) Select “Development Build” to [log debug
      messages](https://docs.unity3d.com/Manual/LogFiles.html).
5. If any scenes are shown in the **Scenes in Build** list, make sure that the
   3DBall Scene is the only one checked. (If the list is empty, then only the
   current scene is included in the build).
6. Click **Build**:
   * In the File dialog, navigate to your ML-Agents directory.
   * Assign a file name and click **Save**.
   * (For Windows）With Unity 2018.1, it will ask you to select a folder instead
     of a file name. Create a subfolder within the ML-Agents folder and select
     that folder to build. In the following steps you will refer to this
     subfolder's name as `env_name`

### 3. Unity ML-Agents Gym Wrapper

A common way in which machine learning researchers interact with simulation
environments is via a wrapper provided by OpenAI called `gym`. For more
information on the gym interface, see [here](https://github.com/openai/gym).

Unity provide a gym wrapper and instructions for using it with existing machine
learning algorithms which utilize gyms. Both wrappers provide interfaces on top
of their `UnityEnvironment` class, which is the default way of interfacing with a
Unity environment via Python.

### Installation

The gym wrapper can be installed using:

```sh
pip install gym_unity
```

or by running the following from the `/gym-unity` directory of the repository:

```sh
pip install .
```

### Using the Gym Wrapper

The gym interface is available from `gym_unity.envs`. To launch an environment
from the root of the project repository use:

```python
from gym_unity.envs import UnityEnv

env = UnityEnv(environment_filename, worker_id, use_visual, uint8_visual, multiagent)
```

*  `environment_filename` refers to the path to the Unity environment.

*  `worker_id` refers to the port to use for communication with the environment.
   Defaults to `0`.

*  `use_visual` refers to whether to use visual observations (True) or vector
   observations (False) as the default observation provided by the `reset` and
   `step` functions. Defaults to `False`.

*  `uint8_visual` refers to whether to output visual observations as `uint8` values 
   (0-255). Many common Gym environments (e.g. Atari) do this. By default they 
   will be floats (0.0-1.0). Defaults to `False`.

*  `multiagent` refers to whether you intent to launch an environment which
   contains more than one agent. Defaults to `False`.

*  `flatten_branched` will flatten a branched discrete action space into a Gym Discrete. 
   Otherwise, it will be converted into a MultiDiscrete. Defaults to `False`.

The returned environment `env` will function as a gym.

### Limitation

* It is only possible to use an environment with a single Brain.
* By default the first visual observation is provided as the `observation`, if
  present. Otherwise vector observations are provided.
* All `BrainInfo` output from the environment can still be accessed from the
  `info` provided by `env.step(action)`.
* Stacked vector observations are not supported.
* Environment registration for use with `gym.make()` is currently not supported.

### 4. Test on python api
After you complete all above setup,
you can use a python api to interact with SoccerTwo Environment.
you also can see ```env.py``` to check detail about that.
Remember your binary file path to replace my binary file path in ```env.py```


### Appendix
The Soccer Environment knowledge... 

### index right version
path : <br>
```linux/soccer_test.x86_64``` <br>
```windows/soccer_twos``` <br>

### Action Space  
#### striker:  
length: 7 <br>
each one dimension: <br>
[0] : not move <br>
[1] : forward <br>
[2] : backward <br>
[3] : rotate Clockwise <br>
[4] : rotate Counterclockwise <br>
[5] : left shift <br>
[6] : right shift <br>

#### goalie :
length :5 <br>
each one dimension: <br>
[0]: not move <br>
[1]: forward <br>
[2]: backward <br>
[3]: left shift <br>
[4]: right shift <br>

### Observation Space
Each striker and each goalie will have 112 dimensions vector which represents their observation. <br>
Vector Observation space: 112 dims corresponding to local 14 ray casts, each detecting 7 possible object types, along with the object's distance. Perception is in a 180-degree view from the front of the agent.<br>

Ray casting: The Ray casting is the ray line that shoots with a specified radius, and angle. In the soccer game setup, the radius is set up to 20 and angle set is [0, 45, 90, 135, 180, 110, 70] <br>

First, they claim they have 14 ray casts, so the angle of ray casts is one of the angles in [0, 45, 90, 135, 180, 110, 70] degrees and they have two height offset[0 1] for observation from the center front of the agent. <br>

Every ray casts will check if they collide with the candidates below shown, then if it collides with any candidates, the dimension of that candidate will become 1. and then append a Distance on the eighth dimension.<br>If it doesn't collide with any candidates, the seventh dimension will become 1.(but the result didn't show the last point I mention but it mention in the source code.)<br>

Example: <br>

[0.         0.         0.         1.         0.         0.         0.         0.21598968 
 <br> 0.         0.         0.         0.         0.         0.         0.         0.         
 <br> 0.         0.         0.         0.         0.         0.         0.         0.
 <br> 0.         0.         0.         0.         0.         0.         0.         0.         
 <br> 0.         0.         0.         1.         0.         0.         0.         0.32554793 
 <br> 0.         0.         0.         0.         0.         0.         0.         0.
 <br> 0.         0.         0.         0.         0.         0.         0.         0.         
 <br> 0.         0.         0.         1.         0.         0.         0.         0.30745506 
 <br> 0.         0.         0.         0.         0.         0.         0.         0.
 <br> 0.         0.         0.         0.         0.         0.         0.         0.         
 <br> 0.         0.         0.         0.         0.         0.         0.         0.        
 <br> 0.         0.         0.         1.         0.         0.         0.         0.40604496
 <br> 0.         0.         0.         0.         0.         0.         0.         0.         
 <br> 0.         0.         0.         0.         0.         0.         0.         0.        ] 
 <br>

 

 The above example is a 112 dimension observation vector for the red agent; If the first row on the fourth dimension is 1, then it means that it is in 0 degrees (left), the ray line collides with the wall and the distance is 0.2159 * 20 = 4.318 (unit).<br>
 [the height offset (from the center front of agent) is 0 at the first 7 rows and the ray horizontally shoots to the height offset 0]<br>
 
 The second rows don't have any value being 1, which means that the rast doesn't collide with any candidate in 45 degrees so the distance is 0.<br>
 The 8th row raises 1 in the fourth dimension when the ray collides with wall and the distance is 0.3057 * 20 = 6.114 unit in 0 degree(left), <br>
[The height offset (from the center front of agent) is 1 at the last 7 rows and the ray shoots to the height offset 0]<br>

Team Blue:
[one_hot_vector]: { ball, red_goal, blue_goal, wall, red_agent, blue_agent} (for red agent) <br>
[one_hot_vector]: { ball, blue_goal, red_goal, wall, blue_agent, red_agent} (for blue agent) <br>
[Distance]: { one scalar / ray_max_distance}

### Reward:
* Agent Reward Function (dependent):
  * Striker:
    * +1 When ball enters opponent's goal.
    * -0.1 When ball enters own team's goal.
    * -0.001 Existential penalty.
  * Goalie:
    * -1 When ball enters team's goal.
    * +0.1 When ball enters opponents goal.
    * +0.001 Existential bonus.

### Index:

<img src='index.png' width=800 height=600> <br>
from left up to right down, the indexes of red agents are 0 - 7. <br>
from left up to right down, the indexes of blue agents are  8 -16. <br>


















