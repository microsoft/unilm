import json  
import os  
from glob import glob


def grit():  
    json_files = glob(f"/path/to/grit/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../grit/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "grit",  
        "weight": 1.0,  
        "name": "grit"  
    }
    
    with open("/path/to/dataset_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
    
if __name__ == "__main__":  
    grit()  