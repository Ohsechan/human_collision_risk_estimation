import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_prefix

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    cap.release()

    return video_length, total_frames

def analyze_videos_in_folder(folder_path):
    video_lengths = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.avi'):
            video_path = os.path.join(folder_path, filename)
            video_length, total_frames = get_video_info(video_path)
            if video_length is not None:
                video_lengths.append(video_length)

    num_videos = len(video_lengths)
    if num_videos == 0:
        return {
            "num_videos": 0,
            "average_length": 0,
            "total_length": 0,
            "video_lengths": []
        }

    average_length = np.mean(video_lengths)
    total_length = np.sum(video_lengths)

    return {
        "num_videos": num_videos,
        "average_length": average_length,
        "total_length": total_length,
        "video_lengths": video_lengths
    }

def analyze_dataset(base_path):
    data = {}
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        for action_class in os.listdir(split_path):
            class_path = os.path.join(split_path, action_class)
            if action_class not in data:
                data[action_class] = {
                    "num_videos": 0,
                    "average_length": 0,
                    "total_length": 0,
                    "total_video_lengths": []
                }
            class_stats = analyze_videos_in_folder(class_path)
            data[action_class]["num_videos"] += class_stats["num_videos"]
            data[action_class]["total_length"] += class_stats["total_length"]
            data[action_class]["total_video_lengths"].extend(class_stats["video_lengths"])

    for action_class in data:
        if data[action_class]["num_videos"] > 0:
            data[action_class]["average_length"] = data[action_class]["total_length"] / data[action_class]["num_videos"]

    return data

def plot_statistics(data):
    categories_order = ["Lying Down", "Sitting", "Sit down", "Stand up", "Standing", "Walking"]
    num_categories = len(categories_order)

    sorted_data = [data[category] for category in categories_order]

    # Plot settings
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    width = 0.25  # 막대그래프 넓이를 줄임

    # Define colors
    colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']

    # Number of videos
    values = [d['num_videos'] for d in sorted_data]
    axes[0].bar(np.arange(num_categories), values, width, color=colors)
    axes[0].set_title('Number of Videos')
    axes[0].set_xticks(np.arange(num_categories))
    axes[0].set_xticklabels(categories_order, rotation=45, ha='right')

    # Average video length
    values = [d['average_length'] for d in sorted_data]
    axes[1].bar(np.arange(num_categories), values, width, color=colors)
    axes[1].set_title('Average Video Length (seconds)')
    axes[1].set_xticks(np.arange(num_categories))
    axes[1].set_xticklabels(categories_order, rotation=45, ha='right')

    # Total video length (sum of all videos)
    total_video_lengths = [sum(d['total_video_lengths']) for d in sorted_data]
    axes[2].bar(np.arange(num_categories), total_video_lengths, width, color=colors)
    axes[2].set_title('Total Video Length (seconds)')
    axes[2].set_xticks(np.arange(num_categories))
    axes[2].set_xticklabels(categories_order, rotation=45, ha='right')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

package_path = find_package_path('image_processing')
dataset_path = os.path.join(package_path, 'dataset_action_split')

# 분석 및 시각화 실행
data = analyze_dataset(dataset_path)
plot_statistics(data)
