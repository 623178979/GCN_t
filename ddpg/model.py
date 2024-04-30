import torch
import torch.nn.functional as F
import torch_geometric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # constraint embedding
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.Linear(cons_nfeats, emb_size, bias=False),
        )

        # self.edge_embedding = torch.nn.Sequential(
        #     torch.nn.LayerNorm(edge_nfeats),
        # )

        # variable emmbedding
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size, bias=False),
        )

        self.conv_c_to_v_1 = BipartiteGraphConvolution()
        self.conv_v_to_c_1 = BipartiteGraphConvolution()

        self.conv_c_to_v_2 = BipartiteGraphConvolution()
        self.conv_v_to_c_2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size,emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size,emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size,1,bias=False),
            torch.nn.Sigmoid()
        )

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
            # print('orivar:',variable_features.size(),'cons:',constraint_features.size(),'edge:',edge_features.size())
            # First step: linear embedding layers to 128 dimension
            variable_features = self.var_embedding(variable_features)
            # edge_features = self.edge_embedding(edge_features)
            constraint_features = self.cons_embedding(constraint_features)
            
            # Two half convolutions
            constraint_features = self.conv_v_to_c_1(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            variable_features = self.conv_c_to_v_1(
                constraint_features, edge_indices, edge_features, variable_features
            )

            # Second convolutions
            constraint_features = self.conv_v_to_c_2(
                variable_features, reversed_edge_indices, edge_features, constraint_features
            )
            # print(constraint_features.size())
            variable_features = self.conv_c_to_v_2(
                constraint_features, edge_indices, edge_features, variable_features
            )

            # A final MLP on the variable features
            output = self.output_module(variable_features).squeeze(-1)
            # print('policyout:',output.size())
            return output

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__(aggr=None)
        emb_size = 128
        
        # self.feature_module_left = torch.nn.Linear(emb_size, emb_size)
        # self.feature_module_edge = torch.nn.Linear(1, emb_size, bias=False)
        # self.feature_module_right = torch.nn.Linear(emb_size, emb_size, bias=False)
        
        # self.feature_module_final = torch.nn.Sequential(
        #     torch.nn.LayerNorm(emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, emb_size),
        # )
        self.feature_module_final = torch.nn.Linear(emb_size,emb_size)

        self.aggregate_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )

        # self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        # self.output_module = torch.nn.Sequential(
        #     torch.nn.Linear(2 * emb_size, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, emb_size),
        # )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        self.adj_matrix = torch.zeros(left_features.shape[0],right_features.shape[0],dtype=torch.float32).to(DEVICE)
        # trans_edge = torch.transpose(edge_features, 0, 1)
        self.adj_matrix[edge_indices[0],edge_indices[1]] = edge_features.T
        # print('convedge:',edge_features.size())
        # self.adj_matrix = torch.transpose(self.adj_matrix, 0, 1)
        # self.left_nodes = left_features
        # self.adj_matrix = edge2adj(edg_indices=edge_indices,
        #                            edge_features=trans_edge,
        #                            left_features_n=left_features.shape[0],
        #                            right_features_n=right_features.shape[0])
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
            right_features=right_features,
        )
        # print('left:',left_features.size(),'edge:',edge_features.size(),'right:',right_features.size())
        # print('forward output:',output.size(),'left:',left_features.size())
        # return self.output_module(
        #     torch.cat([self.post_conv_module(output), right_features], dim=-1)
        # )
        output = left_features + output
        return output
    

    def message(self, node_features_i, node_features_j, right_features):
        # output = self.feature_module_final(
        #     node_features_i
        # )
        # adj_matrix = edge2adj(edg_indices=edge_indices,edge_features=edge_features,left_features_n=node_features_i,right_features_n=node_features_j)
        # print('output:',output.size())
        output = self.feature_module_final(right_features)
        # print('output:',output.size(),'A:',self.adj_matrix.size())
        out = torch.matmul(self.adj_matrix,output)
        return out
    
    def aggregate(self, input):
        out_agg = self.aggregate_module(input)
        return out_agg
