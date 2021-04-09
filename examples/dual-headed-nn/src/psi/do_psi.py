import syft as sy 
import openmined_psi as psi
import torch 
from .psiabs import *
sy.load_lib("openmined_psi")


class PsiProtocolDO(PsiProtocol):
    def __init__(self, duet, dataset, fpr=1e-6, reveal_intersection=True):
        self.duet = duet
        self.dataset = dataset
        self.data_ids = list(map(str, map(int, dataset.ids)))
        self.fpr = fpr
        self.reveal_intersection = reveal_intersection
        
        self.__start_protocol()
        
        self.server = None
        
        
    def __start_protocol(self):
        self.__one_to_one_exchange_init()
        
        
    def __one_to_one_exchange_init(self):
        self.status = PSI_STATUS.INIT
        self.__add_handler_accept(self.duet, tag=PSI_TAGS.REVEAL_INTERSECTION)
        self.__add_handler_accept(self.duet, tag=PSI_TAGS.FPR)
        
        sy_reveal_intersection = sy.lib.python.Bool(self.reveal_intersection)
        sy_reveal_intersection_ptr = sy_reveal_intersection.tag(PSI_TAGS.REVEAL_INTERSECTION).send(self.duet, searchable=True)
        
        sy_fpr = sy.lib.python.Float(self.fpr)
        sy_fpr_ptr = sy_fpr.tag(PSI_TAGS.FPR).send(self.duet, searchable=True)
        
        
    def psi_setup(self):
        self.status = PSI_STATUS.SETUP
        client_items_len = self.__get_object_duet(tag=PSI_TAGS.CLIENT_ITEMS_LEN)
        self.server = psi.server.CreateWithNewKey(self.reveal_intersection)
        setup = self.server.CreateSetupMessage(self.fpr, client_items_len, self.data_ids)
        self.__add_handler_accept(self.duet, tag=PSI_TAGS.SETUP)
        setup_ptr = setup.send(self.duet, searchable=True, tags=[PSI_TAGS.SETUP], description="psi.server Setup Message")
        
        
    def psi_response(self):
        self.status = PSI_STATUS.RESPONSE
        request = self.__get_object_duet(tag=PSI_TAGS.REQUEST)
        response = self.server.ProcessRequest(request) 
        self.__add_handler_accept(self.duet, tag=PSI_TAGS.RESPONSE)
        response_ptr = response.send(self.duet, searchable=True, tags=[PSI_TAGS.RESPONSE], description="psi.server response")      

    def __get_object_duet(self, tag=""):
        return self.duet.store[tag].get(delete_obj=False)
    
    
    def __add_handler_accept(self, duet, action="accept", tag=""):
        duet.requests.add_handler(
                tags=[tag],
                action="accept"
        )

        
class DOPsiStar(PsiProtocolDO):
    def __init__(self, *args, **kwargs): 
        super(DOPsiStar,self).__init__(*args, **kwargs)
        self.id_int_list = None
        
    def retrieve_intersection(self):
        id_int = self.duet.store[PSI_TAGS.IDS_INTERSECT].get(delete_obj=False)
        self.id_int_list = sorted(list(map(int, list(id_int))))
        return self.id_int_list
   
    def share_all_values(self):
        #convert the values to share in tensors
        value_tensor, label_tensor, id_tensor = self.__convert_values_toshare(self.id_int_list)
        
        #share those values
        value_tensor_ptr = value_tensor.send(self.duet, searchable=True, tags=[PSI_TAGS.VALUES], description="intersecting values")
        label_tensor_ptr = label_tensor.send(self.duet, searchable=True, tags=[PSI_TAGS.LABELS], description="intersecting labels")
        id_tensor_ptr = id_tensor.send(self.duet, searchable=True, tags=[PSI_TAGS.IDS], description="intersecting ids") 
        
        self.add_handler_accept(duet, tag=PSI_TAGS.VALUES)
        self.add_handler_accept(duet, tag=PSI_TAGS.LABELS)
        self.add_handler_accept(duet, tag=PSI_TAGS.IDS)
        
    def add_handler_accept(self, duet, action="accept", tag=""):
        duet.requests.add_handler(
                tags=[tag],
                action="accept"
        )
    
    def __convert_values_toshare(self, id_int_list):
        value_list_toshare = []
        label_list_toshare = []
        id_list_toshare = []
        for k in id_int_list:
            if k in self.dataset.values_dic.keys():
                tuple_ = self.dataset.values_dic[k]
                value_list_toshare.append(tuple_[0])
                label_list_toshare.append(tuple_[1])
                id_list_toshare.append(int(k))

        value_tensor = torch.cat(value_list_toshare)
        label_tensor = torch.Tensor(label_list_toshare)
        id_tensor = torch.Tensor(id_list_toshare)
        

        return value_tensor, label_tensor, id_tensor