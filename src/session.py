"""
Session for SAM2-Fuse application. This serves as one "node" in davinci resolve fusion.
Process:
1. User adds node -> creates new session
2. Sends the ONLY the first frame of the video to the session, then it'll wait for points
3. User adds points, then "Start Track" -> session status changes to processing
4. Session processes the video and points, then outputs the masks when done
License: MIT
Author: Robert Zhao
"""
from enum import Enum
from PIL import Image
from sam2 import sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor
from src.configurer import Configurer
import numpy as np
import torch, os, io, base64, threading

configurer = Configurer()

class Point:
    def __init__(self, coord: tuple[int, int], obj_id: int, add: bool = True):
        """
        Creates a point that stores a buncha info.

        :coord: Stores the x and y coord as a tuple
        :obj_id: THe object this point is supposed to track
        :add: The type of point: True = foreground (keep) False = background (exclude)
        """
        self.coord: np.ndarray = np.array([[coord[0], coord[1]]], dtype=np.float32)
        self.obj_id: int = obj_id
        self.type: np.ndarray = np.array([1]) if add else np.array([0])

class Session:
    def __init__(self, session_id: int, model: str) -> None:
        """
        Creates a new session for video processing.

        :session_id: for a unique identification
        :model: a specific sam2 model to use for video segmentation
        :first_frame: takes in the first frame of the clip that the node is added on

        We need to save the session_id, intialize the video propagator, and then put the first frame into frames
        """

        self.session_id: int = session_id
        self.model: str = model
        self.device: str = self._get_device()
        self.predictor = None
        self.state = None
        self.generator = None

        self.directory: str = f"./processing/{session_id}/"

        self.create_video_propagator()
        self._make_directory()

    def _make_directory(self):
        """
        This is required to make the directory full of JPEG files...
        """

        os.makedirs(f"./processing/{str(self.session_id)}/input/", exist_ok=True)
        os.makedirs(f"./processing/{str(self.session_id)}/output/", exist_ok=True)
        

    def _determine_config_file(self) -> str:
        if "large" in self.model:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "base" in self.model:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "small" in self.model:
            return "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "tiny" in self.model:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"
        else:
            return "unknown"
        
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            print("Detected MPS. Going to use it for increased performance. Warning: SAM2 model was trained using CUDA so MPS may give different masks.")
            return "mps"
        else:
            return "cpu"
        
    def create_video_propagator(self):
        """
        Creates the video propagator
        """

        # make sure config exists
        conf_file: str = self._determine_config_file()

        if conf_file == "unknown":
            raise FileNotFoundError("Video propagator file path not determined properly! Please make sure the model names are NOT renamed.")
        
        # determine if the model is installed
        models: dict[str, str] = configurer.get_models()

        # get the path of the config file
        if self.model not in models:
            raise FileNotFoundError("Model passed into session does not exist!")
        
        model_path = models.get(self.model)
        
        self.predictor = build_sam2_video_predictor(conf_file, model_path, self.device)
        self.predictor = self.predictor.to(self.device)
        print("Successfully created the video propagator")

    def add_points(self, frame: int, point: Point) -> None:
        """
        Adds points to the session for later video propagator use. I plan on making this return a preview mask in the future.
        
        :frame: Frame of where the point was added
        :point: The point coords
        """

        self.init_state()

        self.predictor.add_new_points_or_box( # type: ignore
            self.state,
            frame_idx = frame,
            obj_id = point.obj_id,
            points = point.coord,
            labels = point.type
        )
        
    def init_state(self):
        self.state = self.predictor.init_state(self.directory + "input/") # type: ignore

    def propagate(self):
        """
        Will propagate through the session id's processing directory.
        """

        self._make_directory() # make sure directory is made

        frames_count: int = len(os.listdir(f"{self.directory}input/"))

        # make sure we have video state
        self.init_state()

        # regenerate generator
        self.generator = self.predictor.propagate_in_video(self.state) # type: ignore

        for frame_idx in range(frames_count):
            # attempt to get that frame of video
            frame: np.ndarray = np.array([])
            if os.path.exists(f"./processing/{str(self.session_id)}/input/{frame_idx}.jpg"):
                frame = np.array(Image.open(f"./processing/{str(self.session_id)}/input/{frame_idx}.jpg").convert("RGB"))
            else:
                print("Can't get the frame. Is it uploaded?")
                return

            result = None

            # use video propagator
            frame_idx_out, obj_ids, mask_logits = next(self.generator)
            result = (frame_idx_out, obj_ids, mask_logits)

            # create a mask
            mask = (result[2][0] > 0).cpu().numpy().squeeze() # type: ignore

            # combine the masks together
            original = Image.fromarray(frame)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            result = Image.composite(original, Image.new("RGB", original.size, 0), mask_img)

            # instead of returning in the api, we save to disk instead
            result.save(f"{self.directory}output/{frame_idx}.png")