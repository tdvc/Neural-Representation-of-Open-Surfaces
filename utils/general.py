"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
General 

Purpose:
Function used for configuration file - taken from Rasmus Reinhold Paulsen (rapa@dtu.dk)
"""

import os
import datetime
import re
from utils.config import EasyConfig

def parse_cfg_file(cfg_file: str):
    # Load configuration settings from config file
    cfg = EasyConfig()
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg.load(cfg_file, recursive=True)
    return cfg