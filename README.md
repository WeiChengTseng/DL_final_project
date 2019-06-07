# Deep Learning Final Project

## Get Environemnt Binary
Please choose the binary file according to you os. If you want to rebuild the environment, please see this [document](./docs/Readme_rebuild.md).
- MACOS  
For macos, the build binary file is in `./env/macos/SoccerTwosLearner.app`
- Ubuntu  
For Ubuntu, the build binary file is in `./env/linux/soccer_test.x86_64`
- Windows  
For Windows, the build binary file is in `./env/windows/soccer_twos`  

## Development

### Baselines
- PPO
    - developed by Po-Han Chi
    - python main_ppo.py --folder ./PPO/{your name} --rewards_add True --reward_addtion 0.0002
    - python main_ppo.py --folder ./PPO/{your name} --rewards_add False

Model are all saved in ./PPO/
 
- A2C
    - developed by Wei-Cheng Tseng
    - python main_a2c.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --reward_shaping True
    - python main_a2c.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --entropy_coeff 0.02


### Multi-Agent Method
- MADDPG
- MAAC
    - developed by Wei-Cheng Tseng

## Reference

