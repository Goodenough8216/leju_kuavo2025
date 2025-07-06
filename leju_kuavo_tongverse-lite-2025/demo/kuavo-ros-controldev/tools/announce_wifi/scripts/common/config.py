#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

class Config:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self.__dict__.update(json.load(file))
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

if __name__ == '__main__':
    import os
    config_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(config_path, '../../config/config.json')
    config = Config(config_file_path)
    
    print(config.to_json())