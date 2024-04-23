# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/20 4:21
# @Function:


# main.py
from data_process import main as data_process_main
from cluster_analysis import main as cluster_analysis_main
from feature_engineering import main as feature_engineering_main
from train_model import main as train_model_main
from results_interpretation import main as results_interpretation_main
import os


def list_csv_filenames(folder_path):
    """
    This function returns a list of .csv filenames in the given folder.

    Parameters:
    folder_path (str): The path to the folder from which to list .csv filenames.

    Returns:
    list: A list of .csv filenames found in the folder.
    """
    dataset_filenames = {filename[:-4] for filename in os.listdir(folder_path) if filename.endswith('.csv')}
    return dataset_filenames


def main():
    folder_path = '../../dat/analyze/dataset'
    dataset_filenames = list_csv_filenames(folder_path)
    print(dataset_filenames)
    for filename in dataset_filenames:
        print(f"Processing dataset: {filename}")

        print("Starting data processing...")
        data_process_main(filename)

        print("Starting cluster analysis...")
        cluster_analysis_main()

        print("Starting feature engineering...")
        feature_engineering_main()

        print("Starting model training...")
        train_model_main()

        print("Starting results interpretation...")
        results_interpretation_main(filename)

    print("All datasets have been processed.")


if __name__ == '__main__':
    main()
