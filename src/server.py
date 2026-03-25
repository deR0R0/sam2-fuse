"""
Server for SAM2-Fuse for session management.
License: MIT
Author: Robert Zhao
"""

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from uuid import uuid4
from PIL import Image
import numpy as np
import base64
import io
import time
import sys
import os
import signal

from src.configurer import Configurer
from src.session import Session, Point

app = FastAPI()
configurer = Configurer()
last_heartbeat = time.time()

sessions: dict[int, Session] = {}

class New(BaseModel):
    model: str

class AddPoint(BaseModel):
    frame: int
    obj_id: int
    x: int
    y: int
    add: bool

class PropagateNext(BaseModel):
    frame: int
    frame_data: str

def base64_to_numpy(b64: str) -> np.ndarray:
    """
    Change base64 image to numpy array. Also make sure to flip the color channels.
    """
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def shutdown_server():
    os.kill(os.getpid(), signal.SIGTERM)

def check_heartbeat():
    global last_heartbeat
    while True:
        if time.time() - last_heartbeat >= 180: # 3 minutes
            print("Exiting due to inactivity...")
            shutdown_server() # exit to save the user's ram lol
            break

        time.sleep(60)

@app.post("/session/new")
async def new_session(param: New):
    session_id: int = int(uuid4()) % (10 ** 8) # generate a random 8 digit session id

    propa_session = Session(session_id, param.model)

    sessions[session_id] = propa_session

    return {"success": True, "message": "New session created", "session_id": session_id}

@app.post("/session/{id}/point/add")
async def add_point(id: str, param: AddPoint):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    point = Point(
        param.frame,
        (param.x, param.y),
        param.obj_id,
        param.add
    )

    session.add_point(param.frame, point)

    return {"success": True, "message": "Added new point."}

@app.post("/session/{id}/propagate/start")
async def propagate_start(id: str, bg_tasks: BackgroundTasks):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    # make sure the stop propagation is set to false
    session.stop_propagation = False
    
    # send the propagation off to a background task
    bg_tasks.add_task(session.propagate)

    return {"success": True, "message": "Started video propagation"}

@app.post("/session/{id}/propagate/stop")
async def propagate_stop(id: str):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    # stop propagation by setting the reference var to True
    session.stop_propagation = True

    return {"success": True, "message": "Probably stopped video propagation"}

@app.post("/session/{id}/cleanup")
async def cleanup_session(id: str):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    session.cleanup()

    return {"success": True, "message": "Cleaned up some resources"}


@app.get("/session/all")
async def get_all_sessions():
    return {"success": True, "sessions": list(sessions.keys())}


@app.post("/heartbeat/init")
async def heartbeat_init(bg_tasks: BackgroundTasks):
    bg_tasks.add_task(check_heartbeat)
    return {"success": True}

@app.post("/heartbeat/beat")
async def heartbeat_beat():
    global last_heartbeat
    last_heartbeat = time.time()
    return {"success": True}

"""
THE FOLLOWING ENDPOINTS ARE FOR DEBUGGING PURPOSES. THEY WILL NOT BE USED
"""


@app.get("/session/{id}/frames")
def get_frames(id: str):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    propa_session = sessions[session_id]

    return {"success": True, "frames": propa_session.frames.tolist()} # type: ignore