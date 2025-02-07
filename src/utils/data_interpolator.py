import numpy as np
from scipy.interpolate import interp1d
import csv

def interpolate_missing_data(results_csv_path):
    """
    Interpolates missing bounding box data for vehicles and license plates.

    Args:
        results_csv_path (str): Path to the CSV file containing detection results.

    Returns:
        list[dict]: A list of dictionaries with interpolated data for all frames.
    """
    try:
        with open(results_csv_path, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        print(f"Loaded {len(data)} rows from {results_csv_path}.")
    except Exception as e:
        print(f"Error while reading CSV: {e}")
        return []

    frame_numbers = np.array([int(row['frame_nmr']) for row in data], dtype=int)
    car_ids = np.array([int(float(row['car_id'])) for row in data], dtype=int)
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    print(f"Found {len(unique_car_ids)} unique car IDs.")

    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_subset = car_bboxes[car_mask]
        license_plate_bboxes_subset = license_plate_bboxes[car_mask]

        all_frame_numbers = np.arange(car_frame_numbers[0], car_frame_numbers[-1] + 1)

        car_bbox_interpolator = interp1d(car_frame_numbers, car_bboxes_subset, axis=0, kind='linear', fill_value='extrapolate')
        license_bbox_interpolator = interp1d(car_frame_numbers, license_plate_bboxes_subset, axis=0, kind='linear', fill_value='extrapolate')

        interpolated_car_bboxes = car_bbox_interpolator(all_frame_numbers)
        interpolated_license_bboxes = license_bbox_interpolator(all_frame_numbers)

        for i, frame_number in enumerate(all_frame_numbers):
            interpolated_data.append({
                'frame_nmr': int(frame_number),
                'car_id': car_id,
                'car_bbox': list(interpolated_car_bboxes[i]),
                'license_plate_bbox': list(interpolated_license_bboxes[i]),
                'license_plate_bbox_score': '0',
                'license_number': '0',
                'license_number_score': '0'
            })

    return interpolated_data
