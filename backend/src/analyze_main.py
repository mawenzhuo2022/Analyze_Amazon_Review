# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/20 4:21
# @Function:


# main.py
from analyze.data_process import main as data_process_main
from analyze.cluster_analysis import main as cluster_analysis_main
from analyze.feature_engineering import main as feature_engineering_main
from analyze.train_model import main as train_model_main
from analyze.results_interpretation import main as results_interpretation_main


def main():
    print("Starting data processing...")
    data_process_main()

    print("Starting cluster analysis...")
    cluster_analysis_main()

    print("Starting feature engineering...")
    feature_engineering_main()

    print("Starting model training...")
    train_model_main()

    print("Starting results interpretation...")
    results_interpretation_main()

    print("All processes completed.")


if __name__ == '__main__':
    main()
