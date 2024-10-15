import torch
import torch.nn.functional as F
# import torch_geometric
import numpy as np
from ddpg.kan import KANLinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNPolicy(torch.nn.Module):
    def __init__(self,print_ver = False):
        super().__init__()
        self.print_ver = print_ver
        emb_size = 128
        self.emb_size = emb_size
        var_nfeats = 14

        # variable emmbedding
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size, bias=False),
        )
        self.ln_1 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_1 = torch.nn.Linear(emb_size,emb_size)
        self.ln_2 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_2 = torch.nn.Linear(emb_size,emb_size)
        self.ln_3 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_3 = torch.nn.Linear(emb_size,emb_size)
        self.ln_4 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )

        self.kan_1 = KANLinear(emb_size,emb_size,base_activation=torch.nn.SiLU,grid_size=5,spline_order=3)
        self.kan_2 = KANLinear(emb_size,emb_size,base_activation=torch.nn.SiLU,grid_size=5,spline_order=3)
        self.kan_3 = KANLinear(emb_size,1,base_activation=torch.nn.Sigmoid,grid_size=5,spline_order=3)

    def forward(
        self, obs
    ):
            col_num = obs.shape[-1]
            a_mat_shape_1 = obs.shape[1]
            obs_all = obs.reshape(-1,col_num)
            obs_1 = obs_all[:,:14]
            
            a_mat = obs_all[:,-(col_num-14):]
            a_mat = torch.reshape(a_mat,(-1,a_mat_shape_1,col_num-14))
            x = self.var_embedding(obs_1)
            x = torch.reshape(x,(-1,a_mat_shape_1,self.emb_size))
            x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x)
            xc_c = self.ln_1(x_c)
            x_c = self.linear_1(xc_c)
            x_c = torch.matmul(a_mat,x_c)
            x_c = self.ln_2(x_c)

            x_c_v = x_c + x
            x_c = self.linear_2(x_c_v)
            x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x_c)
            x_c = self.ln_3(x_c) + xc_c
            x_c = self.linear_3(x_c)
            x_c = torch.matmul(a_mat,x_c)
            x = self.ln_4(x_c) + x_c_v


            mlp_out = self.kan_1(x)
            mlp_out = self.kan_2(mlp_out)
            mlp_out = self.kan_3(mlp_out)
            out = mlp_out*0.6 + 0.2
            print('kan policy')
            
            return out
    

class GNNCriticmean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 128
        self.emb_size = emb_size
        var_nfeats = 15
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(var_nfeats, emb_size, bias=False),
        )
        self.ln_1 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_1 = torch.nn.Linear(emb_size,emb_size)
        self.ln_2 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_2 = torch.nn.Linear(emb_size,emb_size)
        self.ln_3 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )
        self.linear_3 = torch.nn.Linear(emb_size,emb_size)
        self.ln_4 = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.Tanh(),
        )

        self.final = torch.nn.Linear(emb_size,1)
        self.kan_1 = KANLinear(2*emb_size,2*emb_size,base_activation=torch.nn.SiLU,grid_size=5,spline_order=3)
        self.kan_2 = KANLinear(2*emb_size,emb_size,base_activation=torch.nn.SiLU,grid_size=5,spline_order=3)
        self.kan_3 = KANLinear(emb_size,1,base_activation=torch.nn.Sigmoid,grid_size=5,spline_order=3)

    def forward(self, obs, action):
        # print('obs',obs.shape)
        a_mat_shape_1 = obs.shape[1]
        x_a = torch.tile(action, [1,1,self.emb_size])
        features = obs[:,:,:14]
        a_mat = obs[:,:,-(obs.shape[-1]-14):]
        obs = torch.cat((features, action),dim=-1)
        x = torch.reshape(obs,(-1,15))
        x = self.var_embedding(x)
        x = torch.reshape(x,(-1,a_mat_shape_1,self.emb_size))
        x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x)
        xc_c = self.ln_1(x_c)
        x_c = self.linear_1(xc_c)
        x_c = torch.matmul(a_mat,x_c)
        x_c = self.ln_2(x_c)

        x_c_v = x_c + x
        x_c = self.linear_2(x_c_v)
        x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x_c)
        x_c = self.ln_3(x_c) + xc_c
        x_c = self.linear_3(x_c)
        x_c = torch.matmul(a_mat,x_c)
        x = self.ln_4(x_c) + x_c_v

        x_a_a = x_a * x
        x_a_a = torch.sum(x_a_a,dim=1)/torch.sum(x_a,dim=1)

        x = torch.mean(x, 1)

        x = torch.cat((x, x_a_a), dim=-1)
        x = self.kan_1(x)
        x = self.kan_2(x)
        x = self.kan_3(np.sqrt(2)*x)
        print('kan critic')
        return x



