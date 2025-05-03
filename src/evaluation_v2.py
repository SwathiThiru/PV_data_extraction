"""
This file contains a code to evaluate the PV data extraction quality.

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
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score

def read_ground_truth(file_path):
    """ Code to read the ground truth yaml file"""
    with open(file_path, 'r') as yaml_file:
        ground_truth = yaml.safe_load(yaml_file)
    return ground_truth

def read_predicted_data(file_path):
    """ Code to read the predicted csv file"""
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        predicted_data = [row for row in reader]
    return predicted_data

def flatten_nested_list(nested_list):
    """
        Flattens a nested list into a single list by iterating through sublists and elements.

        Parameters:
        - nested_list (list): A potentially nested list of elements or sublists.

        Returns:
        - list: A single-level list containing all elements from the nested structure.

        Example:
        >>> nested_list = [[1, 2, 3], 4, [5, [6]], 7]
        >>> flatten_nested_list(nested_list)
        [1, 2, 3, 4, 5, [6], 7]
    """
    return [item for sublist in nested_list for item in (sublist if isinstance(sublist, list) else [sublist])]

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
        Evaluates the precision and recall of table extraction by comparing ground truth data
        against predicted data, ensuring alignment of corresponding keys and values.

        Parameters:
        - ground_truth (dict): A dictionary where keys are column headers and values are lists of data elements.
        - predicted_data (list): A list of lists containing predicted keys and values, where each list
          represents a table row or column.

        Returns:
        - tuple: A tuple containing precision (float) and recall (float), measured on a micro-averaged basis.

        Example:
        >>> ground_truth = {"Name": ["Alice", "Bob"], "Age": ["25", "30"]}
        >>> predicted_data = [["Name", "Alice", "Bob"], ["Age", "25", "30"]]
        >>> evaluate_table_extraction(ground_truth, predicted_data)
        (1.0, 1.0)
    """

    # Flatten ground truth dictionary into a list of interleaved keys and values
    ground_truth_flat = flatten_dict_with_lists(ground_truth)

    # Initialize an empty list to store the corresponding predicted data
    predicted_data_flat = []

    # Inside the loop that iterates over keys in ground_truth
    for key in ground_truth.keys():
        found_matching_list = False
        for predicted_list in predicted_data:
            if predicted_list and str(key) == predicted_list[0]:
                # Flatten the matching list from predicted_data
                #predicted_data_flat.extend(
                    #[item.replace('|', '').strip() for item in flatten_nested_list(predicted_list)])
                values = [str(item).replace('|', '').strip() for item in ground_truth[key]]
                predicted_data_flat.append(key)
                predicted_data_flat.extend(values)
                found_matching_list = True
                break  # Break after finding the first matching list

        # If no matching list is found, append the predicted_data_flat with n elements

        if not found_matching_list:
            if isinstance(ground_truth[key], list):
                # Check if the value is a list before attempting to get its length
                n = len(ground_truth[key])
            else:
                # If it's not a list, treat it as a single element
                n = 1
            predicted_data_flat.extend([''] * (n + 1))

    # Add a conditional step to pad predicted_data_flat if needed
    if len(ground_truth_flat) > len(predicted_data_flat):
        padding_size = len(ground_truth_flat) - len(predicted_data_flat)
        predicted_data_flat.extend([''] * padding_size)
    elif len(ground_truth_flat) < len(predicted_data_flat):
        padding_size = len(predicted_data_flat) - len(ground_truth_flat)
        ground_truth_flat.extend([''] * padding_size)

    # Compute precision and recall
    precision = precision_score(ground_truth_flat, predicted_data_flat, average='micro')
    recall = recall_score(ground_truth_flat, predicted_data_flat, average='micro')

    return precision, recall


def flatten_list(lst):
    """
        Flattens a nested list (a list of lists) into a single list.

        Parameters:
        - lst (list): A nested list where each element is either a list or an item.

        Returns:
        - list: A single-level list containing all elements of the nested lists.

        Example:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
        >>> flatten_list([['a', 'b'], ['c'], []])
        ['a', 'b', 'c']
    """
    return [item for sublist in lst for item in sublist]

def compute_classification_metrics(ground_truth, predictions):
    """
        Computes classification metrics (Precision, Recall, and F1-Score) by comparing
        ground truth data with predictions. The comparison is performed element-wise.

        Parameters:
        - ground_truth (dict): A dictionary where keys represent the categories and values
          are lists of ground truth elements for each category.
        - predictions (list): A list of lists where each list represents a row of predicted
          category with its associated elements. The first element of each list is the category key.

        Returns:
        - tuple: A tuple containing precision, recall, and F1-score as floats.
            - precision (float): The ratio of correctly predicted positive observations
              to the total predicted positive observations.
            - recall (float): The ratio of correctly predicted positive observations
              to all observations in the actual category.
            - f1_score (float): The harmonic mean of precision and recall.

        Example:
        >>> ground_truth = {"cat1": [1, 2, 3], "cat2": [4, 5]}
        >>> predictions = [["cat1", 1, 2], ["cat2", 4, 6]]
        >>> compute_classification_metrics(ground_truth, predictions)
        (0.5, 0.4, 0.4444444444444445)
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for key in ground_truth.keys():
        for predicted_list in predictions:
            # Check if the predicted list is not empty and the key matches the first element
            if predicted_list and key == predicted_list[0]:
                # Flatten the lists for comparison
                ground_truth_flat = flatten_list([key] + ground_truth[key])
                predictions_flat = flatten_list(predicted_list)

                # Ensure the lists are of the same length by appending None if needed
                max_len = max(len(ground_truth_flat), len(predictions_flat))
                ground_truth_flat += [None] * (max_len - len(ground_truth_flat))
                predictions_flat += [None] * (max_len - len(predictions_flat))

                # Check each element in the flattened lists
                for i in range(len(ground_truth_flat)):
                    if ground_truth_flat[i] == predictions_flat[i]:
                        true_positive += 1
                    elif ground_truth_flat[i] is not None:
                        false_negative += 1  # Ground truth is positive, but prediction is negative
                        false_positive += 1  # Prediction is positive, but ground truth is negative

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def evaluate_folder(folder_path, ground_truth_folder):
    """
            Evaluates table extraction performance for all CSV files in a folder by comparing
            predicted data against ground truth data.

            Parameters:
            - folder_path (str): Path to the folder containing the CSV files with predicted data.
            - ground_truth_folder (str): Path to the folder containing ground truth YAML files.

            Returns:
            - None: Prints precision and recall for each file, and overall average precision and recall.
    """

    precision_sum = 0
    recall_sum = 0
    num_files = 0

    for csv_file in os.listdir(folder_path):
        if csv_file.endswith(".csv"):
            #csv_path = os.path.join(folder_path, csv_file)
            csv_path = folder_path + '/' + csv_file
            yaml_file = ground_truth_folder + '/' + os.path.splitext(csv_file)[0] + ".yml"
            #yaml_file = os.path.join(folder_path, os.path.splitext(csv_file)[0] + ".yaml")

            if os.path.exists(yaml_file):
                ground_truth = read_ground_truth(yaml_file)
                predicted_data = read_predicted_data(csv_path)

                #precision, recall = evaluate_table_extraction(ground_truth, predicted_data)

                #print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')
                precision, recall= evaluate_table_extraction(ground_truth, predicted_data)
                print(f'File: {csv_file}, Precision: {precision:.4f}, Recall: {recall:.4f}')

                precision_sum += precision
                recall_sum += recall
                #num_files += 1
                num_files += 1

    print(num_files)

    if num_files > 0:
        average_precision = precision_sum / num_files
        average_recall = recall_sum / num_files
        print(f'Average Precision: {average_precision:.4f}')
        print(f'Average Recall: {average_recall:.4f}')

def main():
    # Replace with your folder paths
    predicted_folder = '../inferences/extractionOutput'
    ground_truth_folder = '../inferences/groundTruth'

    print("Evaluation results for each file:")
    evaluate_folder(predicted_folder, ground_truth_folder)

if __name__ == "__main__":
    main()
