import json


def read_json_file(file_path):
    with open(file_path, "r") as file:
        json_data = file.read()
        data_dict = json.loads(json_data)
    return data_dict
