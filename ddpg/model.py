import torch
import torch.nn.functional as F
# import torch_geometric
import numpy as np

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


class GNNPolicy(torch.nn.Module):
    def __init__(self,print_ver = False):
        super().__init__()
        self.print_ver = print_ver
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
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

        # self.edge_embedding = torch.nn.Sequential(
        #     torch.nn.LayerNorm(edge_nfeats),
        # )

        # self.f_lin_1 = torch.nn.Linear(emb_size,emb_size)
        # self.f_lin_2 = torch.nn.Linear(emb_size,emb_size)
        # self.f_lin_3 = torch.nn.Linear(emb_size,1)
        # self.f_tn_1 = torch.nn.Tanh()
        # self.f_tn_2 = torch.nn.Tanh()
        # self.f_sg_1 = torch.nn.Sigmoid()


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size,emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size,emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size,1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, obs
    ):
            col_num = obs.shape[-1]
            print('obsshape',obs.shape)
            # print('colnum',col_num)
            # obs_all = obs.reshape(-1,obs.shape[-1])
            obs_all = obs.reshape(-1,col_num)
            # print('obs_all',obs_all.shape)
            obs_1 = obs_all[:,:14]
            # print('obs_1',obs_1.shape)
            a_mat_shape_1 = obs.shape[1]
            a_mat = obs_all[:,-(col_num-14):]
            # print('a_mat',a_mat.shape)
            # print('a_mat_1',a_mat.shape)
            a_mat = torch.reshape(a_mat,(-1,a_mat_shape_1,col_num-14))
            # print('a_mat_2',a_mat.shape)
            # a_mat = a_mat.reshape(a_mat,(-1,a_mat_shape_1,obs_all.shape[-1]-14))
            # print('orivar:',variable_features.size(),'cons:',constraint_features.size(),'edge:',edge_features.size())
            # First step: linear embedding layers to 128 dimension
            x = self.var_embedding(obs_1)
            # print('emb',x.shape)
            x = torch.reshape(x,(-1,a_mat_shape_1,128))
            # print('reshape_emb',x.shape)
            # print(obs.shape)
            # variable_features = self.var_embedding(variable_features)
            x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x)
            # print('x_c_1',x_c.shape)
            xc_c = self.ln_1(x_c)
            # print('xcc',xc_c.shape)
            # xc_c = torch.nn.Tanh(x_c)
            x_c = self.linear_1(xc_c)
            # print('x_c_2',x_c.shape)
            x_c = torch.matmul(a_mat,x_c)
            # print('x_c_3',x_c.shape)
            x_c = self.ln_2(x_c)
            # print('x_c_4',x_c.shape)

            x_c_v = x_c + x
            # print('x_c_v',x_c_v.shape)
            x_c = self.linear_2(x_c_v)
            # print('x_c_5',x_c.shape)
            x_c = torch.matmul(torch.permute(a_mat,(0,2,1)),x_c)
            # print('x_c_6',x_c.shape)
            x_c = self.ln_3(x_c) + xc_c
            # print('x_c_7',x_c.shape)
            x_c = self.linear_3(x_c)
            # print('x_c_8',x_c.shape)
            x_c = torch.matmul(a_mat,x_c)
            # print('x_c_9',x_c.shape)
            x = self.ln_4(x_c) + x_c_v
            # print('x',x.shape)

            # x = self.f_lin_1(x)
            # x = self.f_tn_1(x)
            # x = self.f_lin_2(x)
            # x = self.f_tn_2(x)
            # x = self.f_lin_3(x)
            # x = self.f_sg_1(x)
            mlp_out = self.output_module(x)
            # print('mlpout',mlp_out.shape)
            out = mlp_out*0.6 + 0.2
            # print('final',out.shape)
            
            return out
    

class GNNCriticmean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 128
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
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size,2*emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(2*emb_size,emb_size),
            # torch.nn.Tanh(),
            # torch.nn.Linear(emb_size,1,bias=False),
            # torch.nn.Sigmoid()
        )
        self.final = torch.nn.Linear(emb_size,1)

    def forward(self, obs, action):
        print('obs',obs.shape)
        a_mat_shape_1 = obs.shape[1]
        x_a = torch.tile(action, [1,1,128])
        features = obs[:,:,:14]
        a_mat = obs[:,:,-(obs.shape[-1]-14):]
        obs = torch.cat((features, action),dim=-1)
        x = torch.reshape(obs,(-1,15))
        x = self.var_embedding(x)
        x = torch.reshape(x,(-1,a_mat_shape_1,128))
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
        x = self.output_module(x)
        x = self.final(np.sqrt(2)*x)

        return x




