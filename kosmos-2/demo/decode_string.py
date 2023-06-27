import re
import numpy as np

def find_patch_index_combinations(s):  
    # The regular expression pattern for matching the required formats  
    pattern = r'(?:(<phrase>([^<]+)</phrase>))?<object>((?:<patch_index_\d+><patch_index_\d+></delimiter_of_multi_objects/>)*<patch_index_\d+><patch_index_\d+>)</object>'  
      
    # Find all matches in the given string  
    matches = re.findall(pattern, s)  
      
    # Initialize an empty list to store the valid patch_index combinations  
    valid_combinations = []  
      
    for match in matches:  
        phrase_tag, phrase, match_content = match  
        if not phrase_tag:  
            phrase = None  
          
        # Split the match_content by the delimiter to get individual patch_index pairs  
        patch_index_pairs = match_content.split('</delimiter_of_multi_objects/>')  
          
        for pair in patch_index_pairs:  
            # Extract the xxxx and yyyy values from the patch_index pair  
            x = re.search(r'<patch_index_(\d+)>', pair)  
            y = re.search(r'<patch_index_(\d+)>', pair[1:])  
              
            if x and y:  
                if phrase:  
                    valid_combinations.append((phrase, int(x.group(1)), int(y.group(1))))  
                else:  
                    valid_combinations.append((f"<{x.group(1)}><{y.group(1)}>", int(x.group(1)), int(y.group(1))))  
      
    return valid_combinations  

def get_box_coords_from_index(P, ul_idx, lr_idx):  
    """  
    Given a grid of length P and the indices of the upper-left and lower-right corners of a bounding box,  
    returns the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2].  
      
    Args:  
    - P (int): the length of the grid  
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box  
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box  
      
    Returns:  
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]  
    """  
    # Compute the size of each cell in the grid  
    cell_size = 1.0 / P  
      
    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box  
    ul_x = ul_idx % P  
    ul_y = ul_idx // P  
      
    lr_x = lr_idx % P  
    lr_y = lr_idx // P  
      
    # Compute the normalized coordinates of the bounding box  
    if ul_idx == lr_idx:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    elif ul_x == lr_x or ul_y == lr_y:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    else:  
        x1 = ul_x * cell_size + cell_size / 2  
        y1 = ul_y * cell_size + cell_size / 2  
        x2 = lr_x * cell_size + cell_size / 2  
        y2 = lr_y * cell_size + cell_size / 2  
      
    return np.array([x1, y1, x2, y2])

def decode_bbox_from_caption(caption, quantized_size=32, **kwargs):
    
    valid_combinations = find_patch_index_combinations(caption)
    entity_names = list(map(lambda x: x[0], valid_combinations))
    patch_index_coords = list(map(lambda pair: get_box_coords_from_index(quantized_size, pair[1], pair[2]), valid_combinations))
    collect_entity_location = []
    for entity_name, patch_index_coord in zip(entity_names, patch_index_coords):
        collect_entity_location.append([entity_name,] + patch_index_coord.tolist())
    
    # print(collect_entity_location)
    return collect_entity_location

if __name__ == "__main__":
    
    caption = "a wet suit is at <object><patch_index_0003><patch_index_0004></delimiter_of_multi_objects/><patch_index_0005><patch_index_0006></object> in the picture" 
    print(find_patch_index_combinations(caption))
    print(decode_bbox_from_caption(caption))

 