from __future__ import print_function
import syft as sy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple
from uuid import UUID
from uuid import uuid4
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler

import datasets


"""This is an experimental work-in-progress feature"""


class EnhanchedWorker():
    """Single worker with a role (label / data holder) and a model"""

    def __init__(self, worker, dataset, model, level=1):
        
        self.worker = worker
        self.dataset = dataset #It can also be None, and then it would be only computational
        self.model = model 

        self.level = max(level, 0) #it should start from zero, otherwise throw error #TODO: implement error throwing



class FederatedWorkerChain():

	"""Class wrapping all the workers with their corresponding model """
	def __init__(self, enhanchedWorkersList):
		self.enhanchedWorkersList = enhanchedWorkersList
		dic_workers = {}
		for ew in enhanchedWorkersList: 
			if ew.level not in dic_workers.keys():
				dic_workers[ew.level] = []

			dic_workers[ew.level].append(ew)

		self.dic_workers = dic_workers


	#TODO: implement check that the level passed is valid
	def get_same_level_en_workers(self, level):
		return self.dic_workers[level]

	def get_previous_level_en_workers(self, level):
		return self.dic_workers[level-1]

	def get_next_level_en_workers(self, level):
		return self.dic_workers[level+1]
