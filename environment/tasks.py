from environment.rumor_env import RumorMillEnv

# ---------- TASK FACTORIES ----------

def task_easy():
    return RumorMillEnv(difficulty=1)

def task_medium():
    return RumorMillEnv(difficulty=2)

def task_hard():
    return RumorMillEnv(difficulty=3)

# ---------- GRADERS ----------

def grade_easy(env: RumorMillEnv):
    return {
        "score": env.social_capital,
        "success": env.social_capital > 80,
    }

def grade_medium(env: RumorMillEnv):
    return {
        "score": env.social_capital,
        "success": env.social_capital > 70,
    }

def grade_hard(env: RumorMillEnv):
    return {
        "score": env.social_capital,
        "success": env.social_capital > 60,
    }