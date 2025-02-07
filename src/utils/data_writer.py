import csv

def write_results_to_csv(results, output_path):
    """
    Write results to a CSV file.
    Args:
        results (list): List of interpolated results (dictionaries).
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'VehicleID', 'CarBBox', 'LicensePlateBBox', 'LicensePlateScore', 'LicenseNumber', 'LicenseNumberScore'])

        for row in results:
            writer.writerow([
                row['frame_nmr'],
                row['car_id'],
                row['car_bbox'],
                row['license_plate_bbox'],
                row['license_plate_bbox_score'],
                row['license_number'],
                row['license_number_score']
            ])
