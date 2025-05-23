import numpy as np

def parse_coords(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                frame, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords.append((frame, x, y))
    return coords

def find_bounce(coords):
    """
    Detects the bounce point as the first local maximum in Y (fall to rise),
    including cases with flat peaks.
    """
    for i in range(1, len(coords) - 2):
        y_prev = coords[i - 1][2]
        y_curr = coords[i][2]
        y_next = coords[i + 1][2]
        y_after = coords[i + 2][2]

        # Case 1: sharp peak
        if y_curr > y_prev and y_curr > y_next:
            return coords[i][0], i

        # Case 2: flat peak
        if y_curr > y_prev and y_curr == y_next and y_next > y_after:
            return coords[i][0], i

    return None, None



def estimate_velocity(coords, dt, bounce_index, window=5):
    start = max(1, bounce_index - window)
    vx_list = []
    vy_list = []
    for i in range(start, bounce_index + 1):
        x0, y0 = coords[i - 1][1], coords[i - 1][2]
        x1, y1 = coords[i][1], coords[i][2]
        vx_list.append((x1 - x0) / dt)
        vy_list.append((y1 - y0) / dt)
    return np.mean(vx_list), np.mean(vy_list)

def simulate_bounce_trajectory(x0, y0, vx, vy, num_frames, dt, gravity=9.8, damping=0.7, drag=0.001):
    points = []
    for frame in range(1, num_frames + 1):
        # Apply physics
        vy += gravity * dt  # gravity acceleration
        vx *= (1 - drag)    # simple air resistance
        vy *= (1 - drag)

        # Update position
        x0 += vx * dt
        y0 += vy * dt

        # If it hits the ground (bottom of frame), bounce
        if y0 >= 1080:
            y0 = 1080
            vy = -vy * damping  # dampened bounce

        points.append((frame, x0, y0))
    return points

def predict_trajectory(coords, bounce_idx, vx, vy, dt, gravity=9.8, damping=0.7, drag=0.001):
    x0, y0 = coords[bounce_idx][1], coords[bounce_idx][2]
    vy = -vy * damping  # apply bounce damping

    remaining_frames = len(coords) - bounce_idx - 1

    return simulate_bounce_trajectory(x0, y0, vx, vy, remaining_frames, dt, gravity, damping, drag)

def save_coords(original, predicted, bounce_frame, filename):
    with open(filename, 'w') as f:
        for frame, x, y in original:
            if frame <= bounce_frame:
                f.write(f"{frame},{int(round(x))},{int(round(y))}\n")
        next_frame = bounce_frame + 1
        for i, (offset, x, y) in enumerate(predicted):
            f.write(f"{next_frame + i},{int(round(x))},{int(round(y))}\n")

def main():
    coords_file = 'coordinates\coordinates.txt'
    output_file = 'coordinates\coordinates_no_spin.txt'

    coords = parse_coords(coords_file)

    fps = 30
    dt = 1 / fps

    bounce_frame, bounce_idx = find_bounce(coords)
    if bounce_frame is None:
        print("Bounce not detected")
        return

    print(f"Bounce detected at frame {bounce_frame}")

    vx, vy = estimate_velocity(coords, dt, bounce_idx)
    print(f"Estimated pre-bounce velocity: vx={vx:.2f}, vy={vy:.2f}")

    predicted = predict_trajectory(coords, bounce_idx, vx, vy, dt)
    save_coords(coords, predicted, bounce_frame, output_file)

    print(f"Prediction saved in {output_file}")

if __name__ == "__main__":
    main()
