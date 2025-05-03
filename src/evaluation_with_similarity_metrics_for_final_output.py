"""
This file contains a code to evaluate the PV data extraction
quality based on the Levenshtein distance to account for OCR errors.

Author:
    Name:
        Swathi Thiruvengadam
    Email:
        swathi.thiruvengadam@ise.fraunhofer.de
        swathi.thiru078@gmail.com
"""

import os
import csv
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
from Levenshtein import distance
import matplotlib.pyplot as plt
from openpyxl import load_workbook

def read_ground_truth(file_path):
    """ Code to read the ground truth yaml file"""
    with open(file_path, 'r', errors='ignore') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(xlsx_path):
    """ Code to read the predicted xlsx file"""
    wb = load_workbook(xlsx_path)
    sheet = wb.active  # or wb[sheet_name] if you know the sheet name
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(row)
    return data

def flatten_dict_with_lists(input_dict):
    """
        Code to flatten a dictionary into a single list. It includes the keys followed by their associated values.
        If a value is a list, its elements are individually added to the flattened list.

        Parameters:
        - input_dict (dict): A dictionary where keys map to either single values or lists of values.

        Returns:
        - list: A flattened list containing keys and their corresponding values.

        Example:
        >>> input_dict = {'a': [1, 2, 3], 'b': 4, 'c': [5, 6]}
        >>> flatten_dict_with_lists(input_dict)
        ['a', 1, 2, 3, 'b', 4, 'c', 5, 6]
        """

    flattened_list = []
    for key, value in input_dict.items():
        flattened_list.append(key)
        if isinstance(value, list):
            flattened_list.extend(value)
        else:
            flattened_list.append(value)
    return flattened_list

def evaluate_table_extraction(ground_truth, predicted_data):
    """
        Evaluates table extraction by comparing ground truth data with predicted data
        and calculating precision, recall, and F1-score.

        Parameters:
        - ground_truth (dict): A dictionary where keys represent categories, and values are lists of expected elements.
        - predicted_data (list): A list of lists where each inner list represents a predicted category
          and its associated predicted elements.

        Returns:
        - tuple: A tuple containing precision, recall, and F1-score as floats.
            - precision (float): Weighted precision score across all categories.
            - recall (float): Micro-average recall score across all categories.
            - f1 (float): Micro-average F1-score across all categories.

        Example:
        >>> ground_truth = {"cat1": ["value1", "value2"], "cat2": ["value3"]}
        >>> predicted_data = [["cat1", "value1", "value2"], ["cat2", "value3"]]
        >>> evaluate_table_extraction(ground_truth, predicted_data)
        (1.0, 1.0, 1.0)
    """

    ground_truth_flat = flatten_dict_with_lists(ground_truth)
    predicted_data_flat = []

    for key in ground_truth.keys():
        found_matching_list = False
        for predicted_list in predicted_data:
            if str(predicted_list[0]) == str(key):
            #if predicted_list and distance(str(key), str(predicted_list[0])) <= 3:
                predicted_data_flat.append(key)
                values = [str(item).replace('|', '').strip() if isinstance(item, (list, tuple)) else str(item).replace('|','').strip() for item in ground_truth[key]]
                predicted_data_flat.extend(predicted_list[1:len(values)+1])
                found_matching_list = True
                break
        if not found_matching_list:
            n = len(ground_truth[key])
            predicted_data_flat.extend([''] * (n + 1))

    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    # Handle NaNs or empty strings
    ground_truth_flat = [str(item) for item in ground_truth_flat]
    predicted_data_flat = [str(item) for item in predicted_data_flat]
    print("Ground Truth Flattened:", ground_truth_flat)
    print("Predicted Data Flattened:", predicted_data_flat)

    precision = precision_score(ground_truth_flat, predicted_data_flat, average='weighted', zero_division=0)
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro', zero_division=0)
    f1 = f1_score(ground_truth_flat, predicted_data_flat, average='micro', zero_division=0)

    return precision, recall, f1

def plot_metrics(file_names, precision_list, recall_list, f1_list):
    """
        Plots precision, recall, and F1 scores for a set of files.

        Parameters:
        - file_names (list of str): List of file names.
        - precision_list (list of float): List of precision scores for each file.
        - recall_list (list of float): List of recall scores for each file.
        - f1_list (list of float): List of F1 scores for each file.

        Returns:
        - None: Displays the plot showing the precision, recall, and F1 score trends for the given files.
    """

    plt.figure(figsize=(10, 6))
    x = range(len(file_names))
    plt.plot(x, precision_list, label='Precision', marker='o')
    plt.plot(x, recall_list, label='Recall', marker='s')
    plt.plot(x, f1_list, label='F1 Score', marker='^')
    plt.xticks(x, file_names, rotation=45, ha='right')
    plt.xlabel('Files')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1 Score for Each File')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_folder(folder_path, ground_truth_folder):
    """
            Evaluates table extraction performance for all CSV files in a folder by comparing
            predicted data against ground truth data.

            Parameters:
            - folder_path (str): Path to the folder containing the CSV files with predicted data.
            - ground_truth_folder (str): Path to the folder containing ground truth YAML files.

            Returns:
            - None: Prints precision and recall for each file, and average overall files along with displaying the plot showing the trends.
    """

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    num_files = 0

    precision_list = []
    recall_list = []
    f1_list = []
    file_names = []

    for xlsx_file in os.listdir(folder_path):
        if xlsx_file.endswith(".xlsx"):
            xlsx_path = os.path.join(folder_path, xlsx_file)
            yaml_file = os.path.join(ground_truth_folder, os.path.splitext(xlsx_file)[0] + ".yml")

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(xlsx_path)

                precision, recall, f1 = evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {xlsx_file}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                if precision != 0 and recall !=0 and f1 != 0:
                    num_files += 1

                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                file_names.append(xlsx_file)

    print('Number of files evaluated:', num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        average_f1 = f1_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')
        print(f'Average F1 Score: {average_f1:.4f}')

    plot_metrics(file_names, precision_list, recall_list, f1_list)

def main():
    predicted_folder = '/app/inferences/pdf'
    ground_truth_folder = '/app/inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()

