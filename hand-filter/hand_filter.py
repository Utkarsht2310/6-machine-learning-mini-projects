import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect both hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Filter functions
def apply_grayscale(image):
    """Convert image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_blur(image):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (21, 21), 0)


def apply_edge_detection(image):
    """Apply Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sepia(image):
    """Apply sepia tone filter"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(image, kernel)


def apply_invert(image):
    """Invert colors"""
    return cv2.bitwise_not(image)


def apply_threshold(image):
    """Apply binary threshold (black and white)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def apply_cartoon(image):
    """Apply cartoon effect"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


# Dictionary of available filters
filters = {
    '1': ('Grayscale', apply_grayscale),
    '2': ('Blur', apply_blur),
    '3': ('Edge Detection', apply_edge_detection),
    '4': ('Sepia', apply_sepia),
    '5': ('Invert', apply_invert),
    '6': ('Black & White', apply_threshold),
    '7': ('Cartoon', apply_cartoon)
}

# Current filter selection
current_filter = '1'


def order_points(pts):
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]  # Bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]  # Top-right (smallest difference)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left (largest difference)

    return rect


def apply_filter_to_region(frame, points, filter_func):
    """
    Apply filter to the region defined by 4 points
    """
    if len(points) != 4:
        return frame

    # Order the points
    rect = order_points(np.array(points, dtype="float32"))

    # Get the bounding box
    x_min = int(max(0, min(rect[:, 0])))
    y_min = int(max(0, min(rect[:, 1])))
    x_max = int(min(frame.shape[1], max(rect[:, 0])))
    y_max = int(min(frame.shape[0], max(rect[:, 1])))

    if x_max <= x_min or y_max <= y_min:
        return frame

    # Create a mask for the quadrilateral
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, rect.astype(np.int32), 255)

    # Extract the region
    roi = frame[y_min:y_max, x_min:x_max].copy()
    mask_roi = mask[y_min:y_max, x_min:x_max]

    # Apply filter to the ROI
    filtered_roi = filter_func(roi)

    # Blend the filtered region back
    result = frame.copy()
    result[y_min:y_max, x_min:x_max] = np.where(
        mask_roi[:, :, np.newaxis] == 255,
        filtered_roi,
        result[y_min:y_max, x_min:x_max]
    )

    return result


def main():
    global current_filter

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Hand Filter Application")
    print("=" * 50)
    print("Instructions:")
    print("- Use your thumb and index finger from BOTH hands")
    print("- These 4 points define the filter area")
    print("- Press number keys (1-7) to change filters:")
    for key, (name, _) in filters.items():
        print(f"  {key}: {name}")
    print("- Press 'q' to quit")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)

        finger_points = []

        # Extract hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks (optional)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get image dimensions
                h, w, _ = frame.shape

                # Extract thumb tip (landmark 4) and index finger tip (landmark 8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Convert normalized coordinates to pixel coordinates
                thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_point = (int(index_tip.x * w), int(index_tip.y * h))

                finger_points.append(thumb_point)
                finger_points.append(index_point)

                # Draw circles on fingertips
                cv2.circle(frame, thumb_point, 10, (0, 255, 0), -1)
                cv2.circle(frame, index_point, 10, (0, 0, 255), -1)

        # If we have 4 points (2 hands detected), apply filter
        if len(finger_points) == 4:
            filter_name, filter_func = filters[current_filter]
            frame = apply_filter_to_region(frame, finger_points, filter_func)

            # Draw the quadrilateral
            pts = order_points(np.array(finger_points, dtype=np.int32))
            cv2.polylines(frame, [pts.astype(np.int32)], True, (255, 255, 0), 2)

        # Display current filter name
        filter_name, _ = filters[current_filter]
        cv2.putText(frame, f"Filter: {filter_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display instruction
        if len(finger_points) < 4:
            cv2.putText(frame, "Show both hands with thumb & index finger",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Hand Filter Application', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in filters:
            current_filter = chr(key)
            print(f"Switched to: {filters[current_filter][0]}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()