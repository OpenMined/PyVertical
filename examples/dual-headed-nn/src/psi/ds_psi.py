import syft as sy 
import openmined_psi as psi
import torch 
from .psiabs import *
sy.load_lib("openmined_psi")


class PsiProtocolDS(PsiProtocol):
    def __init__(self, duet, index_subset):
        self.duet = duet
        self.index_subset = list(map(str,(map(int, index_subset))))
        
        self.reveal_intersection = None
        self.fpr = None
        self.setup = None
        self.client = None
        self.response = None
    
    def psi_init(self):
        self.status = PSI_STATUS.INIT
        self.reveal_intersection = self.__get_object_duet(tag=PSI_TAGS.REVEAL_INTERSECTION)
        self.fpr = self.__get_object_duet(tag=PSI_TAGS.FPR)
        
        sy_client_items_len = sy.lib.python.Int(len(self.index_subset))
        sy_client_items_len_ptr = sy_client_items_len.send(self.duet, searchable=True, tags=[PSI_TAGS.CLIENT_ITEMS_LEN], description="client items length")
    
    def psi_setup(self):
        self.status = PSI_STATUS.SETUP
        
        #wait for 
        self.setup = self.__get_object_duet(tag=PSI_TAGS.SETUP)
        
        #create client request
        self.client = psi.client.CreateWithNewKey(self.reveal_intersection)
        request = self.client.CreateRequest(self.index_subset)
        
        #request
        request_ptr = request.send(self.duet, tags=[PSI_TAGS.REQUEST], searchable=True, description="client request")
        
        
    def psi_response(self):
        self.status = PSI_STATUS.RESPONSE
        self.response = self.__get_object_duet(tag=PSI_TAGS.RESPONSE)
        
        if self.reveal_intersection:
            intersection = self.client.GetIntersection(self.setup, self.response)
            #iset = list(set(intersection))
        self.intersection = intersection
        return intersection
    
    
    def send_intersecting_ids(self): 
        tensor_intersect = torch.Tensor(self.intersection)
        tensor_intersect.send(self.duet,  tags=[PSI_TAGS.IDS_INTERSECT], searchable=True, description="intersection ids")
        
        
    def __get_object_duet(self, tag="", reason=""):
        my_ptr = self.duet.store[tag]
        my_ptr.request(reason=reason,timeout_secs=-1)
        return self.duet.store[tag].get()


class DSPsiStar:
	def __init__(self, duet_list, index_subset):
		assert len(duet_list) > 0,  "At least one duet reference should be provided"
		self.duet_list = duet_list
		self.index_subset = index_subset
        
		self.intersection_list = None

		self.protocol_list = []
		for duet_instance in self.duet_list: 
			self.protocol_list.append(PsiProtocolDS(duet_instance, index_subset))  
		self.__global_init()
        

	def __global_init(self):
		for protocol in self.protocol_list:
			protocol.psi_init()

	def global_setup(self):
		for protocol in self.protocol_list:
			protocol.psi_setup()

	def global_response(self):
		intersection_list = []
		for protocol in self.protocol_list:
			intersection_list.append(protocol.psi_response())
		self.intersection_list = intersection_list

	def global_intersection(self):
		intersection_sets = map(set, self.intersection_list)
		global_int = torch.Tensor(list(set.intersection(*intersection_sets)))
		#I get the id that I have, I reshare now
		for protocol in self.protocol_list:
			protocol.send_intersecting_ids()