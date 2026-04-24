from openenv.core.server import run_server

from environment.rumor_env import RumorMillEnv

def create_env():
    return RumorMillEnv()

def main():
    run_server(env_factory=create_env)


if __name__ == "__main__":
    main()