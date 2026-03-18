"""
Server for SAM2-Fuse for session management.
License: MIT
Author: Robert Zhao
"""

from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
from PIL import Image
import numpy as np
import base64
import io

from src.configurer import Configurer
from src.session import Session, SessionStatus, Point

app = FastAPI()
configurer = Configurer()

sessions: dict[int, Session] = {}

class New(BaseModel):
    initial_image: str
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

@app.post("/session/new")
def new_session(param: New):
    session_id: int = int(uuid4()) % (10 ** 8) # generate a random 8 digit session id

    np_img = base64_to_numpy(param.initial_image)

    propa_session = Session(session_id, param.model, np_img)

    sessions[session_id] = propa_session

    return {"success": True, "message": "New session created", "session_id": session_id}

@app.post("/session/{id}/point/add")
def add_point(id: str, param: AddPoint):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    point = Point(
        (param.x, param.y),
        param.obj_id,
        param.add
    )

    session.add_points(param.frame, point)

    return {"success": True, "message": "Added new point."}

@app.post("/session/{id}/frame/next")
def propagate_next(id: str, param: PropagateNext):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    session = sessions[session_id]

    numpy_img = base64_to_numpy(param.frame_data)

    mask_dict = session.propagate_forward(param.frame, numpy_img)
    
    return {"success": True, "frame_idx": mask_dict["frame_idx"], "mask_b64": mask_dict["mask_b64"]}


@app.get("/session/all")
def get_all_sessions():
    return {"success": True, "sessions": list(sessions.keys())}


"""
THE FOLLOWING ENDPOINTS ARE FOR DEBUGGING PURPOSES. THEY WILL NOT BE USED
"""


@app.get("/session/{id}/frames")
def get_frames(id: str):
    session_id = int(id)
    if session_id not in sessions:
        return {"success": False, "message": "Session not found"}
    
    propa_session = sessions[session_id]

    return {"success": True, "frames": propa_session.frames.tolist()}