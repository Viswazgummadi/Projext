import cv2


def main():
    found_camera = False

    # Try different camera indices
    for i in range(10):  # You can adjust the range as needed
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found_camera = True
            print(f"Camera detected at index {i}")
            break
        cap.release()

    if not found_camera:
        print("No camera detected.")


if __name__ == "__main__":
    main()
