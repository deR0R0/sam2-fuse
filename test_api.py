from PIL import Image
import base64
import io
import httpx
import cv2
import os

session_id = 0

def test_new():
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = httpx.post("http://localhost:8000/session/new", json={"model": "sam2.1_hiera_large.pt"})
    print(response)

def test_video():
    global session_id
    # we need to strip all the image frames from this, encode to base64, then send it to our api
    # then use that base64 to decode and put it all together
    cap = cv2.VideoCapture("./test_video.mp4")

    if not cap.isOpened():
        raise Exception("Error opening video file")
    
    # create new session
    response = httpx.post("http://localhost:8000/session/new", json={"model": "sam2.1_hiera_large.pt"})
    x = response.json()
    
    # save our session id
    session_id = x["session_id"]

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        img = Image.open(io.BytesIO(buffer.tobytes()))
        img.save(f"./processing/{session_id}/input/{frame_count}.jpg")

        frame_count += 1

    cap.release()

    # prompt for x, y coords
    print("x, y, add?")
    point_stuff_idfk = input("")
    point_stuff = point_stuff_idfk.strip().split(", ")

    # add the first point
    add: bool = point_stuff[2].lower() == "y"
    response = httpx.post(f"http://localhost:8000/session/{session_id}/point/add", json={ "frame": 0, "obj_id": 1, "x": int(point_stuff[0]), "y": int(point_stuff[1]), "add": add }, timeout=180)
    print("Response after adding point: ", response)

    if response.status_code == 500:
        print("Stopped early because adding point no work")
        return

    response = httpx.post(f"http://localhost:8000/session/{session_id}/propagate/start")
    print("Response after starting propagation: ", response)

    input("Press enter to stop propagation.")

    response = httpx.post(f"http://localhost:8000/session/{session_id}/propagate/stop")
    print("Response after stopping propagation: ", response)

def test_cleanup():
    global session_id

    response = httpx.post(f"http://localhost:8000/session/{session_id}/cleanup")
    print("Response after cleanup: ", response)

def test_delete():
    global session_id


    response = httpx.post(f"http://localhost:8000/session/{session_id}/delete")
    print("Response after deleting: ", response)


if __name__ == "__main__":
    print("[0] Exit")
    print("[1] Test New Session")
    print("[2] Test Video")
    print("[3] Test CleanUp")
    print("[4] Test Delete")
    while True:
        x = input("")
        match x:
            case "0":
                exit()
            case "1":
                test_new()
            case "2":
                test_video()
            case "3":
                test_cleanup()
            case "4":
                test_delete()