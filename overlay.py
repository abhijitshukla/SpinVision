import cv2
import numpy as np

def read_coordinates(file_path):
    coords = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                frame, x, y = map(int, line.strip().split(','))
                coords[frame] = (x, y)
    return coords

def find_bounce_frame(coords):
    """
    Detects the bounce frame by identifying the first local maximum in Y,
    including flat peaks (y[i] == y[i+1] > y[i+2]).
    """
    coords_sorted = sorted(coords.items())
    for i in range(1, len(coords_sorted) - 2):
        _, (_, y_prev) = coords_sorted[i - 1]
        frame_curr, (_, y_curr) = coords_sorted[i]
        _, (_, y_next) = coords_sorted[i + 1]
        _, (_, y_after) = coords_sorted[i + 2]

        # Case 1: clear peak
        if y_curr > y_prev and y_curr > y_next:
            return frame_curr

        # Case 2: flat peak
        if y_curr > y_prev and y_curr == y_next and y_next > y_after:
            return frame_curr

    return None


def draw_quadrant(frame, origin, size=80, thickness=2):
    x, y = origin
    color = (255, 255, 0)  # Cyan
    # X and Y axes
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "+X", (x + size + 5, y - 5), font, 0.5, color, 1)
    cv2.putText(frame, "-X", (x - size - 25, y - 5), font, 0.5, color, 1)
    cv2.putText(frame, "-Y", (x + 5, y + size + 15), font, 0.5, color, 1)
    cv2.putText(frame, "+Y", (x + 5, y - size - 10), font, 0.5, color, 1)

def calculate_spin_angle(original_coords, predicted_coords, bounce_frame):
    original_list = sorted(original_coords.items())
    predicted_list = sorted(predicted_coords.items())

    bounce_pos = original_coords[bounce_frame]
    end_actual = original_list[-1][1]
    end_pred = predicted_list[-1][1]

    v_actual = np.array([end_actual[0] - bounce_pos[0], end_actual[1] - bounce_pos[1]])
    v_pred = np.array([end_pred[0] - bounce_pos[0], end_pred[1] - bounce_pos[1]])

    norm_actual = np.linalg.norm(v_actual)
    norm_pred = np.linalg.norm(v_pred)

    if norm_actual == 0 or norm_pred == 0:
        return 0.0

    cos_theta = np.dot(v_actual, v_pred) / (norm_actual * norm_pred)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)

def main():
    video_path = "videos\\kohli.mp4"
    output_path = "output\overlay_output.mp4"

    original_coords = read_coordinates("coordinates\coordinates.txt")
    predicted_coords = read_coordinates("coordinates\coordinates_no_spin.txt")

    bounce_frame = find_bounce_frame(original_coords)
    if bounce_frame is None:
        print("Bounce frame not found.")
        return
    bounce_pos = original_coords[bounce_frame]

    spin_angle = calculate_spin_angle(original_coords, predicted_coords, bounce_frame)
    print(f"Spin angle deviation: {spin_angle:.2f}°")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    green_trail = []
    red_trail = []
    snapshot_frame = None

    last_green_frame = max(original_coords.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num in original_coords:
            green_trail.append(original_coords[frame_num])
        if frame_num in predicted_coords:
            red_trail.append(predicted_coords[frame_num])

        # Draw trajectory lines
        if len(green_trail) > 1:
            for i in range(1, len(green_trail)):
                cv2.line(frame, green_trail[i - 1], green_trail[i], (0, 255, 0), 2)
        if len(red_trail) > 1:
            for i in range(1, len(red_trail)):
                cv2.line(frame, red_trail[i - 1], red_trail[i], (0, 0, 255), 2)

        # Draw trajectory dots
        for pt in green_trail:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
        for pt in red_trail:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        # Draw current positions
        if frame_num in original_coords:
            cv2.circle(frame, original_coords[frame_num], 15, (0, 255, 0), 3)
        if frame_num in predicted_coords:
            cv2.circle(frame, predicted_coords[frame_num], 10, (0, 0, 255), 2)

        # Draw quadrant after bounce
        if frame_num >= bounce_frame:
            draw_quadrant(frame, bounce_pos)

        # Legends
        cv2.putText(frame, "Original (Spin): Green", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Predicted (No Spin): Red", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show spin angle after bounce
        if frame_num >= bounce_frame:
            cv2.putText(frame, f"Spin Angle Deviation: {spin_angle:.2f} degrees",
                        (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

        # Save snapshot when green trajectory completes
        if frame_num == last_green_frame:
            snapshot_frame = frame.copy()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if snapshot_frame is not None:
        cv2.imwrite("output\snapshot_with_spin_angle.png", snapshot_frame)
        print("Snapshot image saved as snapshot_with_spin_angle.png")

    print(f"Output video saved as {output_path}")

if __name__ == "__main__":
    main()
