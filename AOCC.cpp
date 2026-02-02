/*
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
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <iomanip>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct Event {
    double t;
    int x;
    int y;
    int p;
};

void write_to_csv(const std::string& file_path, const std::vector<std::string>& data) {
    std::ofstream file(file_path, std::ios::app);
    if (file.is_open()) {
        for (size_t i = 0; i < data.size(); ++i) {
            file << data[i];
            if (i < data.size() - 1) file << ",";
        }
        file << "\n";
        file.close();
    }
}

std::vector<Event> read_events_from_txt(const std::string& file_path) {
    std::vector<Event> events;
    std::ifstream file(file_path);
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double t;
        int x, y, p;
        int label = -1;
        
        iss >> x >> y >> p >> t;
        // iss >> t >> x >> y >> p;
        if (iss >> label) {
            if (label == 1) {
                events.push_back({t, x, y, p});
            }
        } else {
            events.push_back({t, x, y, p});
        }
    }
    
    return events;
}

std::vector<cv::Mat> create_accumulation_images(const std::vector<Event>& events, int width, int height, double interval) {
    std::vector<cv::Mat> frames;
    cv::Mat current_frame = cv::Mat::zeros(height, width, CV_64F);
    double last_time = -1;
    
    for (const auto& event : events) {
        if (last_time < 0) {
            last_time = event.t;
        }
        
        while (event.t - last_time >= interval) {
            if (cv::countNonZero(current_frame) > 0) {
                frames.push_back(current_frame.clone());
            }
            current_frame = cv::Mat::zeros(height, width, CV_64F);
            last_time += interval;
        }
        
        if (event.y >= 0 && event.y < height && event.x >= 0 && event.x < width) {
            current_frame.at<double>(event.y, event.x) = 1;
        }
    }
    
    if (cv::countNonZero(current_frame) > 0) {
        frames.push_back(current_frame);
    }
    
    return frames;
}

cv::Mat apply_gaussian_blur(const cv::Mat& image, int kernel_size = 5, double sigma = 2) {
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(kernel_size, kernel_size), sigma);
    return blurred;
}

double calculate_contrast(const cv::Mat& image) {
    cv::Mat normalized;
    
    if (image.type() != CV_8U) {
        double min_val, max_val;
        cv::minMaxLoc(image, &min_val, &max_val);
        
        if (max_val == min_val) {
            normalized = cv::Mat::zeros(image.size(), CV_8U);
        } else {
            cv::Mat temp;
            image.convertTo(temp, CV_64F);
            temp = (temp - min_val) / (max_val - min_val) * 255;
            temp.convertTo(normalized, CV_8U);
        }
    } else {
        normalized = image.clone();
    }
    
    cv::Mat gray;
    if (normalized.channels() == 3) {
        cv::cvtColor(normalized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = normalized;
    }
    
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(magnitude, mean, stddev);
    
    return stddev[0];
}

std::tuple<double, double, double> compute_statistics(const std::vector<double>& contrasts) {
    if (contrasts.empty()) {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
    
    double mean_val = std::accumulate(contrasts.begin(), contrasts.end(), 0.0) / contrasts.size();
    
    std::vector<double> sorted_contrasts = contrasts;
    std::sort(sorted_contrasts.begin(), sorted_contrasts.end());
    double median_val;
    size_t n = sorted_contrasts.size();
    if (n % 2 == 0) {
        median_val = (sorted_contrasts[n/2 - 1] + sorted_contrasts[n/2]) / 2.0;
    } else {
        median_val = sorted_contrasts[n/2];
    }
    
    double sum_sq = 0.0;
    for (double c : contrasts) {
        sum_sq += c * c;
    }
    double rms_val = std::sqrt(sum_sq / contrasts.size());
    
    return std::make_tuple(mean_val, median_val, rms_val);
}

struct CSVData {
    std::vector<double> interval_us;
    std::vector<double> mean_contrast;
};

CSVData read_csv(const std::string& csv_path) {
    CSVData data;
    std::ifstream file(csv_path);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stod(token));
        }
        
        if (values.size() >= 2) {
            data.interval_us.push_back(values[0]);
            data.mean_contrast.push_back(values[1]);
        }
    }
    
    return data;
}

double calculate_area_under_curve(const std::string& output_csv, const std::string& csv_path,
                                   double x_min, double x_max, const std::string& pure_filename) {
    CSVData data = read_csv(csv_path);
    
    std::vector<std::pair<double, double>> filtered;
    for (size_t i = 0; i < data.interval_us.size(); ++i) {
        if (data.interval_us[i] >= x_min && data.interval_us[i] <= x_max) {
            filtered.push_back({data.interval_us[i], data.mean_contrast[i]});
        }
    }
    
    std::sort(filtered.begin(), filtered.end());
    
    // Trapezoidal rule
    double area = 0.0;
    for (size_t i = 1; i < filtered.size(); ++i) {
        double dx = filtered[i].first - filtered[i-1].first;
        double avg_y = (filtered[i].second + filtered[i-1].second) / 2.0;
        area += dx * avg_y;
    }
    
    std::ofstream file(output_csv, std::ios::app);
    if (file.is_open()) {
        file << pure_filename << "," << area << "\n";
        file.close();
    }
    
    return area;
}

void plot_single_ccc_with_aocc(const std::string& csv_path, const std::string& output_dir,
                                double min_val, double max_val, const std::string& pure_filename) {
    CSVData data = read_csv(csv_path);
    
    if (data.interval_us.empty()) return;
    
    double x_min_data = *std::min_element(data.interval_us.begin(), data.interval_us.end());
    double x_max_data = *std::max_element(data.interval_us.begin(), data.interval_us.end());
    double y_min_data = *std::min_element(data.mean_contrast.begin(), data.mean_contrast.end());
    double y_max_data = *std::max_element(data.mean_contrast.begin(), data.mean_contrast.end());
    
    int img_width = 800;
    int img_height = 400;
    int margin_left = 80;
    int margin_right = 40;
    int margin_top = 40;
    int margin_bottom = 60;
    
    cv::Mat plot_img(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    auto map_x = [&](double x) -> int {
        return margin_left + (int)((x - x_min_data) / (x_max_data - x_min_data) * (img_width - margin_left - margin_right));
    };
    auto map_y = [&](double y) -> int {
        return img_height - margin_bottom - (int)((y - y_min_data) / (y_max_data - y_min_data + 1e-9) * (img_height - margin_top - margin_bottom));
    };
    
    // Draw grid
    cv::line(plot_img, cv::Point(margin_left, margin_top), cv::Point(margin_left, img_height - margin_bottom), cv::Scalar(0, 0, 0), 1);
    cv::line(plot_img, cv::Point(margin_left, img_height - margin_bottom), cv::Point(img_width - margin_right, img_height - margin_bottom), cv::Scalar(0, 0, 0), 1);
    
    // Draw full CCC curve (blue)
    for (size_t i = 1; i < data.interval_us.size(); ++i) {
        cv::line(plot_img,
                 cv::Point(map_x(data.interval_us[i-1]), map_y(data.mean_contrast[i-1])),
                 cv::Point(map_x(data.interval_us[i]), map_y(data.mean_contrast[i])),
                 cv::Scalar(255, 0, 0), 1);
    }
    
    // Draw AOCC segment (red) and fill (orange)
    std::vector<cv::Point> aocc_points;
    for (size_t i = 0; i < data.interval_us.size(); ++i) {
        if (data.interval_us[i] >= min_val && data.interval_us[i] <= max_val) {
            aocc_points.push_back(cv::Point(map_x(data.interval_us[i]), map_y(data.mean_contrast[i])));
        }
    }
    
    if (!aocc_points.empty()) {
        // Fill area
        std::vector<cv::Point> fill_points = aocc_points;
        fill_points.push_back(cv::Point(aocc_points.back().x, img_height - margin_bottom));
        fill_points.push_back(cv::Point(aocc_points.front().x, img_height - margin_bottom));
        cv::fillPoly(plot_img, std::vector<std::vector<cv::Point>>{fill_points}, cv::Scalar(0, 165, 255));
        
        // Draw red line
        for (size_t i = 1; i < aocc_points.size(); ++i) {
            cv::line(plot_img, aocc_points[i-1], aocc_points[i], cv::Scalar(0, 0, 255), 2);
        }
    }
    
    // Title and labels
    cv::putText(plot_img, "CCC Curve for " + pure_filename, cv::Point(margin_left, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(plot_img, "Interval (us)", cv::Point(img_width/2 - 40, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    
    cv::Mat rotated_label(100, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(rotated_label, "Mean Contrast", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
    
    std::string out_fig = output_dir + "/" + pure_filename + "_ccc_curve.png";
    cv::imwrite(out_fig, plot_img);
}

void process_single_file(const std::string& txt_file_path, int width, int height,
                         const std::string& results_csv_dir, double min_interval, double max_interval,
                         double step, const std::string& save_directory, double min_value, double max_value,
                         bool save_images, const std::string& image_output_dir) {
    
    fs::path p(txt_file_path);
    std::string base_name = p.stem().string();
    std::string ccc_csv_path = results_csv_dir + "/" + base_name + "_ccc.csv";
    
    // Write header
    {
        std::ofstream file(ccc_csv_path);
        file << "Interval (us),Mean Contrast,Median Contrast,RMS Contrast\n";
        file.close();
    }
    
    std::vector<Event> events = read_events_from_txt(txt_file_path);
    std::cout << "Processing file: " << base_name << std::endl;
    std::cout << "File: " << base_name << " | Events: " << events.size() 
          << " | First T: " << (events.empty() ? 0 : events[0].t) 
          << " | Last T: " << (events.empty() ? 0 : events.back().t) << std::endl;
    int total_steps = (int)((max_interval - min_interval) / step);
    int current_step = 0;
    
    for (double interval = min_interval; interval < max_interval; interval += step) {
        current_step++;
        std::cout << "\rProcessing " << base_name << ": " << current_step << "/" << total_steps << std::flush;
        
        std::vector<cv::Mat> accumulation_images = create_accumulation_images(events, width, height, interval);
        
        double mean_contrast = 0.0, median_contrast = 0.0, rms_contrast = 0.0;
        
        if (!accumulation_images.empty()) {
            std::vector<cv::Mat> blurred_frames;
            for (const auto& frame : accumulation_images) {
                blurred_frames.push_back(apply_gaussian_blur(frame, 5, 2));
            }
            
            std::vector<double> contrasts;
            for (const auto& frame : blurred_frames) {
                contrasts.push_back(calculate_contrast(frame));
            }
            
            std::tie(mean_contrast, median_contrast, rms_contrast) = compute_statistics(contrasts);
            
            // Save 3rd frame (index 2)
            if (save_images && !image_output_dir.empty() && blurred_frames.size() >= 3) {
                cv::Mat third_frame = blurred_frames[2];
                
                double min_val_img, max_val_img;
                cv::minMaxLoc(third_frame, &min_val_img, &max_val_img);
                
                cv::Mat normalized;
                if (max_val_img == min_val_img) {
                    normalized = cv::Mat::zeros(third_frame.size(), CV_8U);
                } else {
                    cv::Mat temp;
                    third_frame.convertTo(temp, CV_64F);
                    temp = (temp - min_val_img) / (max_val_img - min_val_img) * 255;
                    temp.convertTo(normalized, CV_8U);
                }
                
                cv::Mat img_mapped(normalized.size(), CV_8U, cv::Scalar(255));
                
                for (int r = 0; r < normalized.rows; ++r) {
                    for (int c = 0; c < normalized.cols; ++c) {
                        uchar val = normalized.at<uchar>(r, c);
                        if (val > 0) {
                            img_mapped.at<uchar>(r, c) = 255 - val;
                        }
                    }
                }
                
                std::string interval_dir = image_output_dir + "/" + base_name;
                fs::create_directories(interval_dir);
                std::string out_path = interval_dir + "/interval_" + std::to_string((int)interval) + "us_frame2.png";
                cv::imwrite(out_path, img_mapped);
            }
        }
        
        std::ofstream file(ccc_csv_path, std::ios::app);
        file << std::fixed << std::setprecision(6);
        file << interval << "," << mean_contrast << "," << median_contrast << "," << rms_contrast << "\n";
        file.close();
    }
    std::cout << std::endl;
    
    // Calculate AOCC
    double area = calculate_area_under_curve(save_directory, ccc_csv_path, min_value, max_value, base_name);
    std::cout << base_name << ". AOCC: " << area << std::endl;
    
    // Plot CCC curve
    if (save_images && !image_output_dir.empty()) {
        plot_single_ccc_with_aocc(ccc_csv_path, image_output_dir, min_value, max_value, base_name);
    }
}

int main() {
    // --- Configuration parameters ---
    std::string input_folder = "";
    std::string results_csv_path = "/exist";
    std::string save_directory = "/exist.csv";
    std::string image_output_dir = "/exist_images";
    
    int width = 1280, height = 720;
    double min_interval = 4000;
    double max_interval = 50001;
    double step = 1000;
    double min_value = 0;
    double max_value = max_interval - 1;
    
    // --- Create directories ---
    fs::create_directories(results_csv_path);
    fs::create_directories(image_output_dir);
    
    // --- Initialize AOCC summary file ---
    {
        std::ofstream file(save_directory);
        file << "Filename,Area Under Curve\n";
        file.close();
    }
    
    // --- Get all .txt files ---
    std::vector<std::string> txt_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".txt") {
            txt_files.push_back(entry.path().filename().string());
        }
    }
    std::sort(txt_files.begin(), txt_files.end());
    
    if (txt_files.empty()) {
        std::cout << "Warning: No .txt files found in " << input_folder << std::endl;
        return 0;
    }
    
    std::cout << "Found " << txt_files.size() << " .txt files to process." << std::endl;
    
    // --- Process files ---
    for (const auto& txt_file : txt_files) {
        std::string full_path = input_folder + "/" + txt_file;
        process_single_file(
            full_path,
            width,
            height,
            results_csv_path,
            min_interval,
            max_interval,
            step,
            save_directory,
            min_value,
            max_value,
            true,
            image_output_dir
        );
    }
    
    std::cout << "All data saved to '" << save_directory << "' and '" << results_csv_path << "'." << std::endl;
    std::cout << "Accumulation images and CCC curves saved to '" << image_output_dir << "'." << std::endl;
    
    return 0;
}

//g++ -Wall -Wextra -g3 AOCC-TFD.cpp -o /home/ps/DOCKER/denoise/TFD/AOCC-TFD     $(pkg-config --cflags --libs opencv4)
