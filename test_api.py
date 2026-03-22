from PIL import Image
import base64
import io
import httpx
import cv2
import os


def test_new():
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = httpx.post("http://localhost:8000/session/new", json={"initial_image": b64, "model": "sam2.1_hiera_large.pt"})
    print(response)

def test_video():
    # we need to strip all the image frames from this, encode to base64, then send it to our api
    # then use that base64 to decode and put it all together
    cap = cv2.VideoCapture("./1000014181.mp4")

    if not cap.isOpened():
        raise Exception("Error opening video file")

    base64_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(frame_base64)

        frame_count += 1

    cap.release()

    # after processing, we need to now send it to our api one by one

    # create new session
    response = httpx.post("http://localhost:8000/session/new", json={"initial_image": base64_frames[0], "model": "sam2.1_hiera_large.pt"})
    x = response.json()
    
    # save our session id
    session_id = x["session_id"]

    # prompt for x, y coords
    print("x, y, add?")
    point_stuff_idfk = input("")
    point_stuff = point_stuff_idfk.strip().split(", ")

    # add the first point
    add: bool = point_stuff[2].lower() == "y"
    response = httpx.post(f"http://localhost:8000/session/{session_id}/point/add", json={ "frame": 0, "obj_id": 1, "x": int(point_stuff[0]), "y": int(point_stuff[1]), "add": add })
    print("Response after adding point: ", response)

    if response.status_code == 500:
        print("Stopped early because adding point no work")
        return

    for frame in base64_frames:
        response = httpx.post(f"http://localhost:8000/session/{session_id}/frame/next", json={ "frame": base64_frames.index(frame), "frame_data": base64_frames[base64_frames.index(frame)] }, timeout=60)
        print("Response after propagating: ", response)
        os.makedirs("./test_output/", exist_ok=True)
        response = response.json()
        img_bytes = base64.b64decode(response["mask_b64"])
        img = Image.open(io.BytesIO(img_bytes))
        img.save(f"./test_output/{base64_frames.index(frame)}.jpg")


if __name__ == "__main__":
    print("[0] Exit")
    print("[1] Test New Session")
    print("[2] Test Video")
    while True:
        x = input("")
        match x:
            case "0":
                exit()
            case "1":
                test_new()
            case "2":
                test_video()