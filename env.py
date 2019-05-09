from gym_unity.envs import UnityEnv
env_name = "./env/Soccer"
env = UnityEnv(env_name, worker_id=0, use_visual=True)