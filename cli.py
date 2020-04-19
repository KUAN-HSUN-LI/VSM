import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", "-r", action="store_true")
    parser.add_argument("--query_file", "-i", required=True, type=str)
    parser.add_argument("--ranked_list", "-o", required=True, type=str)
    parser.add_argument("--model_dir", "-m", required=True, type=folder_path)
    parser.add_argument("--NTCIR_dir", "-d", required=True, type=folder_path)
    return parser.parse_args()


def folder_path(path):
    return Path(path)
