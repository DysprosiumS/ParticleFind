import cv2
import csv
import argparse
import logging
import time
from pathlib import Path
from ultralytics import YOLO

def combined_detection(self, frame):
    """Комбинированный подход: YOLO + OpenCV + фильтрация"""
    all_detections = []

    # 1. YOLO детекция
    yolo_detections = self.detect_with_preprocessing(frame)
    all_detections.extend(yolo_detections)

    # 2. OpenCV детекция
    opencv_detections = self.detect_circles_opencv(frame)
    all_detections.extend(opencv_detections)

    # 3. Удаление дубликатов
    filtered_detections = self.remove_duplicates(all_detections)

    return filtered_detections


def remove_duplicates(self, detections):
    """Удаление дублирующихся детекций"""
    if not detections:
        return []

    # Сортируем по уверенности
    detections.sort(key=lambda x: x['conf'], reverse=True)

    filtered = []
    for det in detections:
        is_duplicate = False
        for existing in filtered:
            # Проверяем пересечение по IoU
            iou = self.calculate_iou(det, existing)
            if iou > 0.3:  # Если пересекаются более чем на 30%
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(det)

    return filtered


def calculate_iou(self, det1, det2):
    """Вычисление Intersection over Union"""
    x1_1, y1_1 = det1['x'] - det1['w'] / 2, det1['y'] - det1['h'] / 2
    x1_2, y1_2 = det1['x'] + det1['w'] / 2, det1['y'] + det1['h'] / 2

    x2_1, y2_1 = det2['x'] - det2['w'] / 2, det2['y'] - det2['h'] / 2
    x2_2, y2_2 = det2['x'] + det2['w'] / 2, det2['y'] + det2['h'] / 2

    # Вычисление площади пересечения
    xi1 = max(x1_1, x2_1)
    yi1 = max(y1_1, y2_1)
    xi2 = min(x1_2, x2_2)
    yi2 = min(y1_2, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Вычисление площадей
    area1 = det1['w'] * det1['h']
    area2 = det2['w'] * det2['h']
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0