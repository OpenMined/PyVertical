from abc import ABC, abstractmethod


class PSI_TAGS:
    REVEAL_INTERSECTION = "reveal_intersection"
    FPR = "fpr"
    CLIENT_ITEMS_LEN = "client_items_len"
    SETUP = "setup"
    REQUEST = "request"
    RESPONSE = "response"
    VALUES = "values"
    LABELS = "labels"
    IDS = "ids"
    IDS_INTERSECT = "ids_intersec"
    
    
class PSI_STATUS: 
    PRE = 0
    INIT = 1
    SETUP = 2
    RESPONSE = 3

class PsiProtocol(ABC):
    def __init__(self, duet):
        self.duet = duet
        self.status = PSI_STATUS.PRE
    
    @abstractmethod
    def psi_setup(self):
        pass
    
    @abstractmethod
    def psi_response(self):
        pass