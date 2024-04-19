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


dataset_filenames = {
    "iphone7 (refurbished)", "iphone7", "iphone11",
    "iphone14", "onePlus7T", "oppoA5",
    "redmiA8", "redmiNote8", "redmiNote9",
    "redmiNote9pro", "redmiNote9promax", "samsungM01",
    "samsungM21", "samsungM31", "samsungZflip",
    "tecno", "vivo"
}

def main():
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
