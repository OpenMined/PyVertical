from .datasets import BaseIndexSet, SampleIndexSetWithLabels
import torch
import syft as sy
from abc import ABC, abstractmethod
import pickle
from .psi_cls.psiabs import PsiProtocol
from .psi_cls.ds_psi import DSPsiStar
from .psi_cls.ds_psi import PsiProtocolDS


class DuetRole(ABC): 
    def __init__(self, data_dir=None, label_dir=None, index_dir=None, name=""):
        
        self.name = name
        
        #load data
        self.index_list = self.__load_data(index_dir) 
        data_list = self.__load_data(data_dir) 
        label_list = self.__load_data(label_dir) 
        
        #create dataset
        self.dataset = self.__create_dataset(data_list, label_list, self.index_list)
        
        self.duet = None
        self.protocol = None
        
    def __load_data(self, directory):
        return pickle.load(open(directory, 'rb'), encoding='utf-8') if directory else None
    
    def __create_dataset(self, data_list, label_list, index_list):
        if not index_list: 
            label_set = torch.Tensor(label_list) if label_list else None
            bs = torch.Tensor(data_list) if data_list else None
            if label_set != None and bs != None:
                dataset = SampleSetWithLabelsNoIndex(label_set, bs)
            else:
                dataset = bs if bs else label_set

        else: #there is index
            label_set = BaseIndexSet(index_list, self.name, label_list, is_labels=True) if label_list else None
            bs = BaseIndexSet(index_list, self.name, data_list, is_labels=False) if data_list else None
            if label_set != None and bs != None:
                dataset = SampleIndexSetWithLabels(label_set, bs, self.name)
            else:
                dataset = bs if bs else label_set
            
        return dataset
        
    @abstractmethod
    def connect_to_duet(self, loopback=True, server_id=None):
        pass
    

class DataScientist(DuetRole):
    def __init__(self, *args, **kwargs): 
        super(DataScientist,self).__init__(*args, **kwargs)
        self.duet_list = []
        self.do_names = []
        self.dic_owner_duet = {}
    
    def connect_to_duet(self, loopback=True, server_id=None, name=None):
        duet = sy.join_duet(loopback=loopback) if loopback else sy.join_duet(server_id)
        self.duet_list.append(duet)
        if name: 
            self.do_names.append(name)
            self.dic_owner_duet[name] = duet
        return duet

    def set_protocol(self, psi_prot, name=None):
        if psi_prot == DSPsiStar:
            self.protocol = psi_prot(self.duet_list, self.index_list)
        if psi_prot == PsiProtocolDS:
            chosen_duet =  self.dic_owner_duet[name] if name else self.duet_list[0]
            self.protocol = psi_prot(chosen_duet, self.index_list)

        
class DataOwner(DuetRole):    
    def connect_to_duet(self, loopback=True):
        self.duet = sy.launch_duet(loopback=loopback)
        return self.duet 

    def set_protocol(self, psi_prot):
        self.protocol = psi_prot(self.duet, self.dataset)
