import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        
        self.fca1 = nn.Linear(192,256)
        self.fca2 = nn.Linear(256,128)
        self.fcb1 = nn.Linear(512,256)
        self.fcb2 = nn.Linear(256,128)
        self.fc = nn.Linear(256,256)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        
    def forward(self, x1, x2):
        
        x1 = self.fca1(x1)
        x1 = F.relu(x1)
        x1 = self.dp1(x1)
        x1 = self.fca2(x1)
        x1 = F.relu(x1)  
        
        x2 = self.fcb1(x2)
        x2 = F.relu(x2)
        x2 = self.dp2(x2)
        x2 = self.fcb2(x2)
        x2 = F.relu(x2)    
        
        x = torch.cat([x1, x2], dim = 1)
        x = self.fc(x)
        x = F.relu(x)
    
        return x

