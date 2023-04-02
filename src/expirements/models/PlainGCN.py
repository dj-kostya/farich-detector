from torch_geometric import nn
from torch.nn import Module, ReLU, Dropout,Linear, Sigmoid
class PlainGCN(Module):

    def __init__(self, **kwargs):
        super(PlainGCN, self).__init__()


        self.num_features = kwargs["num_features"] \
            if "num_features" in kwargs.keys() else 3

        self.num_classes = kwargs["num_classes"] \
            if "num_classes" in kwargs.keys() else 2

        # hidden layer node features
        self.hidden = 256

        self.model = nn.Sequential("x, edge_index, batch_index", [            
                (nn.GCNConv(self.num_features, self.hidden), 
                    "x, edge_index -> x1"),
                (ReLU(), "x1 -> x1a"),                                       
                (Dropout(p=0.5), "x1a -> x1d"),                              
                (nn.GCNConv(self.hidden, self.hidden), "x1d, edge_index -> x2"), 
                (ReLU(), "x2 -> x2a"),                                         
                (Dropout(p=0.5), "x2a -> x2d"),                                
                (nn.GCNConv(self.hidden, self.hidden), "x2d, edge_index -> x3"), 
                (ReLU(), "x3 -> x3a"),                                         
                (Dropout(p=0.5), "x3a -> x3d"),                               
                (nn.GCNConv(self.hidden, self.hidden), "x3d, edge_index -> x4"), 
                (ReLU(), "x4 -> x4a"),                                         
                (Dropout(p=0.5), "x4a -> x4d"),                                
                (nn.GCNConv(self.hidden, self.hidden), "x4d, edge_index -> x5"), 
                (ReLU(), "x5 -> x5a"),                                   
                (Dropout(p=0.5), "x5a -> x5d"),                                
                (nn.global_mean_pool, "x5d, batch_index -> x6"),                  
                (Linear(self.hidden, self.num_classes), "x6 -> x_out"),
                (Sigmoid(), "x_out -> x_class"),
                ])    
       
    def forward(self, x, edge_index, batch):
        x_out = self.model(x, edge_index, batch)

        return x_out