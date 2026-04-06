from fastapi import FastAPI
from env.environment import DeliveryEnv

app = FastAPI()

# create environment
env = DeliveryEnv()

# ✅ ROOT (VERY IMPORTANT — avoids 404)
@app.get("/")
def home():
    return {"status": "running"}

# ✅ RESET
@app.get("/reset")
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": str(state)}

# ✅ STEP
@app.get("/step")
@app.post("/step")
def step(action: str = "move_A"):
    state, reward, done, _ = env.step(action)
    return {
        "state": str(state),
        "reward": reward,
        "done": done
    }

# ✅ STATE
@app.get("/state")
def get_state():
    return {"state": str(env.state())}