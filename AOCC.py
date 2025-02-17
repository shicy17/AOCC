"""
Copyright (C) 2025 Beihang University, Neuromorphic Vision Perception and Computing Group

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright © Beihang University, Neuromorphic Vision Perception and Computing Group.
License: This code is licensed under the GNU General Public License v3.0.
You can redistribute it and/or modify it under the terms of the GPL-3.0 License.
"""

import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
import os
import csv


def write_to_csv(file_path, data):
    """
    Append a row of data to a CSV file.

    Args:
        file_path (str): Path to the CSV file
        data (list): Row of data to be written
    """
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def read_events_from_txt(file_path):
    """
    Read event camera data from a text file.
    Each line contains: timestamp x-coordinate y-coordinate polarity

    Args:
        file_path (str): Path to the text file containing event data

    Yields:
        tuple: (timestamp, x, y, polarity) for each event
    """
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 4:
                t = float(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                p = int(parts[3])
                yield (t, x, y, p)


def create_accumulation_images(events, width, height, interval):
    """
    Create time-based accumulation frames from event data.

    Args:
        events (list): List of event tuples (t, x, y, p)
        width (int): Width of the output image
        height (int): Height of the output image
        interval (float): Time interval for accumulating events

    Returns:
        list: List of accumulated event frames
    """
    frames = []
    current_frame = np.zeros((height, width), dtype=np.uint8)
    last_time = None
    frame_count = 0

    for t, x, y, p in events:
        if last_time is None:
            last_time = t

        while t - last_time >= interval:
            if np.any(current_frame):
                frames.append(current_frame.copy())
                frame_count += 1
            current_frame = np.zeros((height, width), dtype=np.uint8)
            last_time += interval

        current_frame[y, x] = 255

    if np.any(current_frame):
        frames.append(current_frame)

    return frames


def apply_gaussian_blur(image, kernel_size=5, sigma=2):
    """
    Apply Gaussian blur to an image.

    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian kernel

    Returns:
        numpy.ndarray: Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def calculate_contrast(image):
    """
    Calculate image contrast using Sobel operators.

    Args:
        image (numpy.ndarray): Input image (grayscale or BGR)

    Returns:
        float: Standard deviation of gradient magnitudes
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    contrast = np.std(magnitude)

    return contrast


def compute_statistics(contrasts):
    """
    Compute statistical measures from contrast values.

    Args:
        contrasts (list): List of contrast values

    Returns:
        tuple: (mean, median, root mean square) of contrast values
    """
    mean_val = np.mean(contrasts)
    median_val = np.median(contrasts)
    rms_val = np.sqrt(np.mean(np.square(contrasts)))
    return mean_val, median_val, rms_val


def calculate_area_under_curve(output_csv, csv_path, y_column, x_column, x_min, x_max, pure_filename):
    """
    Calculate the area under the curve within specified x-axis bounds.

    Args:
        output_csv (str): Path to save the results
        csv_path (str): Path to input CSV file
        y_column (str): Name of the y-axis column
        x_column (str): Name of the x-axis column
        x_min (float): Minimum x value for integration
        x_max (float): Maximum x value for integration
        pure_filename (str): Filename without extension

    Returns:
        float: Area under the curve
    """
    # Read CSV file
    data = pd.read_csv(csv_path)

    # Filter data based on x-axis range
    filtered_data = data[(data[x_column] >= x_min) & (data[x_column] <= x_max)]

    # Sort data by x-axis values
    filtered_sorted = filtered_data.sort_values(by=x_column)

    # Get x and y values
    x_values = filtered_sorted[x_column].values
    y_values = filtered_sorted[y_column].values

    # Calculate area using trapezoidal rule
    area = trapezoid(y_values, x_values)
    with open(output_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pure_filename, area])
    return area


def plot_contrast_statistics(csv_path_0, csv_path, figure_name, min_value, max_value):
    """
    Plot contrast statistics for multiple CSV files and calculate areas under curves.

    Args:
        csv_path_0 (str): Path to save area results
        csv_path (str): Directory containing input CSV files
        figure_name (str): Name of the output figure
        min_value (float): Minimum x value for area calculation
        max_value (float): Maximum x value for area calculation
    """
    plt.figure(figsize=(10, 6))
    for filename in os.listdir(csv_path):
        if filename.endswith('.csv'):
            pure_filename = os.path.splitext(filename)[0]
            data = pd.read_csv(csv_path + '/' + filename)
            plt.plot(data['Interval (us)'].iloc[:200], data['Mean Contrast'].iloc[:200],
                     label=pure_filename, marker='o', markersize=1)
            area = calculate_area_under_curve(csv_path_0, csv_path + '/' + filename,
                                              'Mean Contrast', 'Interval (us)',
                                              min_value, max_value, pure_filename)
            print(pure_filename, area)

    plt.title('Contrast Statistics Over Different Intervals')
    plt.xlabel('Interval (us)')
    plt.ylabel('Contrast Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{figure_name}')
    plt.show()


def main():
    """
    Main function to process event camera data and calculate contrast statistics.
    """
    # Set file paths and image dimensions
    file_path = './f171hz_fla.txt'  # Event data file path
    width, height = 346, 260  # Resolution of event cameras
    results_csv_path = './202523' # Path for saving CCC data
    min_interval = 2000  # Minimum interval for plotting CCC (us)
    max_interval = 200001  # Maximum interval for plotting CCC (us)
    step = 2000  # Step size for plotting CCC (us)
    ccc_csv_path = results_csv_path + '/f171hzfla_noise.csv' # CCC.csv file name
    save_directory = './202523/result.csv' # AOCC results
    figure_name = 'A.png' # ccc curve
    min_value = 0 # lower bound of ccc
    max_value = max_interval-1 # upper bound of ccc

    # Create save directory if it doesn't exist
    if not os.path.exists(results_csv_path):
        os.makedirs(results_csv_path)

    # Initialize results CSV with headers
    with open(ccc_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Interval (us)', 'Mean Contrast', 'Median Contrast', 'RMS Contrast'])

    results = []

    # Calculate Continuous Contrast Curve (CCC)
    for interval in tqdm.tqdm(np.arange(min_interval, max_interval, step)):
        events = list(read_events_from_txt(file_path))
        accumulation_images = create_accumulation_images(events, width, height, interval)
        contrasts = [calculate_contrast(apply_gaussian_blur(frame, 5, 2)) for frame in accumulation_images]
        mean_contrast, median_contrast, rms_contrast = compute_statistics(contrasts)
        results = (interval, mean_contrast, median_contrast, rms_contrast)
        write_to_csv(ccc_csv_path, results)

    # Calculate Area of CCC (AOCC)
    plot_contrast_statistics(save_directory, results_csv_path, figure_name, min_value, max_value)

    print(f"Data saved to '{save_directory}'.")


if "__main__":
    main()