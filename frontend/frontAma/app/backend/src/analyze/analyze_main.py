# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/4/20 4:21
# @Function:


# main.py
from .data_process import main as data_process_main
from .cluster_analysis import main as cluster_analysis_main
from .feature_engineering import main as feature_engineering_main
from .train_model import main as train_model_main
from .results_interpretation import main as results_interpretation_main
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


def save_file(file_obj, target_directory):
    print("Current working directory:", os.getcwd())  # 打印当前工作目录
    abs_target_directory = os.path.abspath(target_directory)
    print("Absolute target directory:", abs_target_directory)  # 打印绝对路径目录

    if not os.path.exists(abs_target_directory):
        os.makedirs(abs_target_directory)
        print(f"Directory {abs_target_directory} created")  # 目录创建成功的信息

    file_path = os.path.join(abs_target_directory, file_obj.name)
    print("File path to save:", file_path)  # 打印文件存储路径

    with open(file_path, 'wb+') as destination:
        if hasattr(file_obj, 'chunks'):
            for chunk in file_obj.chunks():
                destination.write(chunk)
            print("File written with chunks")
        else:
            destination.write(file_obj.read())
            print("File written with read")

    return file_path

def main(file_obj=None):

    if file_obj:
        save_file(file_obj, 'app/backend/dat/analyze/test')
        file =file_obj.name[:-4]
        print(file)


        print(f"Processing dataset: {file}")

        print("Starting data processing...")
        data_process_main(file)

        print("Starting cluster analysis...")
        cluster_analysis_main()

        print("Starting feature engineering...")
        feature_engineering_main()

        print("Starting model training...")
        train_model_main()

        print("Starting results interpretation...")
        result = results_interpretation_main(file)

        print("All datasets have been processed.")
        return result  # 返回结果

if __name__ == '__main__':
    main()
