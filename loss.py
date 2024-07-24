'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class Softmax(nn.Module):
	def __init__(self, nOut, **kwargs):
		super(Softmax, self).__init__()	    
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut, 2)

	def forward(self, x, label=None):
		x 		= self.fc(x)
		if label != None:
			nloss   = self.criterion(x, label)
			prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]
			return nloss, prec1
		else:
			predScore = x[:,0]
			predScore = predScore.t()
			predScore = predScore.view(-1).detach().cpu().numpy()
			return predScore
		
