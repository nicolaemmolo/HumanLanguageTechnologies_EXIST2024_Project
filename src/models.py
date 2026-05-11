import torch
import torch.nn as nn
from transformers import AutoModel

# Model based on a pretrained transformer (XLM-RoBERTa) with a custom feedforward head
class MultiLabelClassifier(nn.Module):
    def __init__(self, pretrained_name, dropout, hidden_layer_1_size, hidden_layer_2_size, transf_out_size):
        super().__init__()

        self.pretrained = AutoModel.from_pretrained(pretrained_name)
        
        # custom head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transf_out_size, hidden_layer_1_size),
            nn.GELU(),
            nn.Linear(hidden_layer_1_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 6),
            nn.Sigmoid()
        )

    # method that freezes the module weights, preventing them from being updated
    def freeze_pretrained(self):
        self.pretrained.requires_grad_(False)
    
    def unfreeze_pretrained(self):
        self.pretrained.requires_grad_(True)
        
    def forward(self, x):
        embedding = self.pretrained(**x)[0][:,0,:]
        
        return self.head(embedding)
    
# Classifier Chain model architecture (based on XLM-RoBERTa)
class MultiLabelClassifierChain(nn.Module):
    def __init__(self, pretrained_name, dropout, hidden_layer_1_size, hidden_layer_2_size, transf_out_size):
        super().__init__()

        self.pretrained = AutoModel.from_pretrained(pretrained_name)
        
        # dropout layer before classifier head
        self.dropout_layer = nn.Dropout(dropout)
        
        # custom head
        self.parent_head = nn.Sequential(
            nn.Linear(transf_out_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 1),
            nn.Sigmoid()
        )
        self.child_head_segment_1 = nn.Sequential(
            nn.Linear(transf_out_size, hidden_layer_1_size),
            nn.GELU()
        )
        self.child_head_segment_2  = nn.Sequential(
            nn.Linear(hidden_layer_1_size + 1, hidden_layer_2_size),
            nn.GELU(),
            nn.Linear(hidden_layer_2_size, 5),
            nn.Sigmoid()
        )

    # method that freezes the module weights, preventing them from being updated
    def freeze_pretrained(self):
        self.pretrained.requires_grad_(False)
    
    def unfreeze_pretrained(self):
        self.pretrained.requires_grad_(True)
         
    def forward(self, x, *labels):
        tweet_embedding = self.pretrained(**x)[0][:,0,:]
        dropout_output = self.dropout_layer(tweet_embedding)
        
        parent_output = self.parent_head(dropout_output)
        child_output_segment_1 = self.child_head_segment_1(dropout_output)
        
        if self.training:     
            child_input_segment_2 = torch.cat((child_output_segment_1, labels[0]), 1)
            child_output = self.child_head_segment_2(child_input_segment_2)

        else:
            child_input_segment_2 = torch.cat((child_output_segment_1, parent_output), 1)
            child_output = self.child_head_segment_2(child_input_segment_2)

        return torch.cat((parent_output,child_output),1)