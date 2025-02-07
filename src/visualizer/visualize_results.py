import cv2
import pandas as pd
import ast

def draw_bbox(frame, bbox, color, label=None):
    """Draw bounding box and optional label on frame."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def visualize_results(video_path, results_csv, output_video_path):
    """Visualize bounding boxes and license plate text on video."""
    data = pd.read_csv(results_csv)
    cap = cv2.VideoCapture(video_path)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw bounding boxes for the current frame
        frame_data = data[data['frame_nmr'] == frame_count]
        for _, row in frame_data.iterrows():
            car_bbox = ast.literal_eval(row['car_bbox'])
            lp_bbox = ast.literal_eval(row['license_plate_bbox'])
            license_text = row.get('license_number', '')

            draw_bbox(frame, car_bbox, (0, 255, 0), label="Car")
            draw_bbox(frame, lp_bbox, (0, 0, 255), label=license_text)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Visualization complete. Saved to {output_video_path}")

if __name__ == "__main__":
    visualize_results(
        video_path="data/sample.mp4",
        results_csv="data/output/test_interpolated.csv",
        output_video_path="data/output/output_video.mp4"
    )
