# virtual_keyboard.py
import cv2                                   # 1
import mediapipe as mp                       # 2
import time                                 # 3
import numpy as np                          # 4
import pyautogui                             # 5
from math import hypot                       # 6

# -------- CONFIG --------
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720       # 7
SMOOTHING = 5                                # 8  # higher -> smoother cursor, but more lag
CLICK_DISTANCE = 40                          # 9  # pixels threshold for pinch (click)
CLICK_COOLDOWN = 0.6                         # 10 # seconds between accepted clicks to avoid repeats

# -------- SETUP MEDIAPIPE HANDS --------
mp_hands = mp.solutions.hands                # 11
mp_drawing = mp.solutions.drawing_utils      # 12
hands = mp_hands.Hands(min_detection_confidence=0.7,  # 13
                       min_tracking_confidence=0.6,
                       max_num_hands=1)      # only one hand for simplicity

# -------- KEYBOARD LAYOUT --------
# design a simple QWERTY-like single-row or multi-row keyboard displayed on screen.
keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"],
        ["SPACE"]]                           # 14

# compute button sizes based on frame size
button_w = 100                               # 15
button_h = 80                                # 16
start_x = 50                                 # 17
start_y = FRAME_HEIGHT - (len(keys) * (button_h + 10)) - 20  # 18

# helper state
prev_x, prev_y = 0, 0                        # 19  # previous smoothed cursor pos
last_click_time = 0                          # 20

# Start camera
cap = cv2.VideoCapture(0)                    # 21
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)   # 22
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) # 23

# main loop
while True:                                  # 24
    success, frame = cap.read()              # 25
    if not success:                          # 26
        print("Ignoring empty camera frame.")# 27
        time.sleep(0.1)                      # 28
        continue                             # 29

    # mirror the frame so it feels like a mirror (left-right)
    frame = cv2.flip(frame, 1)               # 30
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 31

    # process with MediaPipe
    results = hands.process(frame_rgb)       # 32

    # draw keyboard (behind hand so keys remain visible)
    overlay = frame.copy()                   # 33
    key_positions = []                       # 34  # store rectangles and labels
    y = start_y                              # 35
    for row in keys:                         # 36
        x = start_x                          # 37
        for key in row:                      # 38
            # draw rounded rectangle like button
            cv2.rectangle(overlay, (x, y), (x + button_w, y + button_h), (50, 50, 50), -1) # 39
            cv2.putText(overlay, key, (x + 20, y + button_h//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2) # 40
            key_positions.append((key, (x, y, x + button_w, y + button_h))) # 41
            x += button_w + 10               # 42
        y += button_h + 15                   # 43

    # alpha blend overlay into frame for nicer look
    alpha = 0.6                              # 44
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # 45

    finger_x = None                           # 46  # cursor candidate x (in pixels)
    finger_y = None                           # 47
    clicked = False                           # 48

    if results.multi_hand_landmarks:          # 49
        hand_landmarks = results.multi_hand_landmarks[0]  # 50

        # Get normalized landmark coords and convert to pixel coords
        h, w, _ = frame.shape                 # 51
        # index finger tip is landmark 8, thumb tip is 4, middle tip is 12 (MediaPipe indexing)
        idx_tip = hand_landmarks.landmark[8]  # 52
        thumb_tip = hand_landmarks.landmark[4]# 53
        mid_tip = hand_landmarks.landmark[12]# 54

        # convert normalized to pixel coordinates (note mirrored frame)
        ix, iy = int(idx_tip.x * w), int(idx_tip.y * h) # 55
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h) # 56
        mx, my = int(mid_tip.x * w), int(mid_tip.y * h) # 57

        # smoothing cursor movement: simple moving average like approach
        if prev_x == 0 and prev_y == 0:      # 58
            sx, sy = ix, iy                 # 59
        else:
            sx = prev_x + (ix - prev_x) / SMOOTHING  # 60
            sy = prev_y + (iy - prev_y) / SMOOTHING  # 61

        finger_x, finger_y = int(sx), int(sy)   # 62
        prev_x, prev_y = sx, sy                 # 63

        # draw cursor
        cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), cv2.FILLED) # 64

        # draw tips for debugging/visual feedback
        cv2.circle(frame, (ix, iy), 6, (255, 0, 0), cv2.FILLED)  # index actual
        cv2.circle(frame, (tx, ty), 6, (0, 255, 0), cv2.FILLED)  # thumb actual

        # calculate distance between index tip and thumb tip
        dist_thumb_index = hypot(ix - tx, iy - ty)  # 65

        # detect click if distance below threshold and cooldown passed
        if dist_thumb_index < CLICK_DISTANCE and (time.time() - last_click_time) > CLICK_COOLDOWN:  # 66
            clicked = True
            last_click_time = time.time()

        # OPTIONAL: You can also use index-middle pinch as alternative:
        # dist_index_mid = hypot(ix - mx, iy - my)

        # draw hand landmarks for clarity
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # 67

    # if we have a cursor, check which key it's over
    if finger_x is not None and finger_y is not None:  # 68
        for key, (x1, y1, x2, y2) in key_positions:    # 69
            if x1 < finger_x < x2 and y1 < finger_y < y2:  # 70
                # highlight the hovered key
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 3)  # 71
                cv2.putText(frame, key, (x1 + 20, y1 + button_h//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,140,255), 3)  # 72
                if clicked:  # 73
                    # show visual click feedback
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)  # 74
                    cv2.putText(frame, key, (x1 + 20, y1 + button_h//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)  # 75

                    # actually send the keypress to OS
                    if key == "SPACE":  # 76
                        pyautogui.press("space")  # 77
                    else:
                        pyautogui.press(key.lower())  # 78

                    # small delay to allow visual to render (not required but nicer)
                    time.sleep(0.08)  # 79
                break  # 80

    # show FPS on frame
    cv2.putText(frame, f"Press 'q' to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)  # 81
    cv2.imshow("Virtual Keyboard", frame)       # 82

    key = cv2.waitKey(1) & 0xFF                 # 83
    if key == ord('q'):                         # 84
        break                                   # 85

# cleanup
cap.release()                                  # 86
cv2.destroyAllWindows()                        # 87
hands.close()                                  # 88
