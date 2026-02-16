from info_gather_field import InfoGatherField
from typing import List

class InfoBook:
    info: List[InfoGatherField]
    
    def __init__(self, info: List[InfoGatherField]):
        self.info = info