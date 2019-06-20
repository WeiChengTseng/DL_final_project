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
    - python main_maac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu
    - python main_maac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 1e5
    - python main_maac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 100000 --steps_per_update 1000 --attend_heads 2
    - python main_maac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 100000  --attend_heads 1
    - python main_maac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 100000  --attend_heads 2 --duplicate_policy True --num_updates 2
    - python main_maac_double.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 100000  --attend_heads 2  --num_updates 2
    - python main_maac_ac.py --env_path ./env/linux/SoccerTwosBirdView.x86_64 --steps_per_update 9600 --episode_length 300 --use_gpu --buffer_length 100000  --attend_heads 2  --num_updates 4
## Reference

