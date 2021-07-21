import io, requests
import torch
import torch.nn as nn

from dall_e.encoder import Encoder
from dall_e.decoder import Decoder
from dall_e.utils   import map_pixels, unmap_pixels

def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
            
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)
