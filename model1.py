import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from torch.autograd import Function


class Embedding(nn.Module):
    def __init__(self, d_model, num_of_hour, nodes, num_of_features, type):
        super(Embedding, self).__init__()
        self.num_of_hour = num_of_hour
        self.nodes = nodes
        self.num_of_features = num_of_features
        self.type = type
        self.pos_TEmbedding = nn.Embedding(num_of_hour, nodes)
        self.pos_SEmbedding = nn.Embedding(nodes, num_of_hour)
        self.Tnorm = nn.LayerNorm(nodes)
        self.Snorm = nn.LayerNorm(num_of_hour)

    def forward(self, x, batch_size):
        """
        :param x: B, F, N, T
        :param batch_size:
        :return:
        """
        x = x.squeeze(1)
        if self.type == 'T':
            pos = torch.arange(self.num_of_hour).cuda()
            pos = pos.unsqueeze(0).expand(batch_size, self.num_of_hour)
            pos_embedding = self.pos_TEmbedding(pos)  # B, T, N
            Embedding = x.permute(0, 2, 1) + pos_embedding
            Emx = self.Tnorm(Embedding).permute(0, 2, 1).unsqueeze(1)
        else:
            pos = torch.arange(self.nodes).cuda()
            pos = pos.unsqueeze(0).expand(batch_size, self.nodes)
            pos_embedding = self.pos_SEmbedding(pos)
            Embedding = x + pos_embedding
            Emx = self.Snorm(Embedding).unsqueeze(1)
        return Emx


class Pos_Input(nn.Module):
    def __init__(self, d_model, nodes, num_of_hour, num_of_features):
        super(Pos_Input, self).__init__()
        self.TEmbedding = Embedding(d_model, num_of_hour, nodes, num_of_features, 'T')
        self.SEmbedding = Embedding(d_model, num_of_hour, nodes, num_of_features, 'S')

    def forward(self, x, batch_size):
        """
        :param x: B, F, N, T
        :param batch_size:
        :return:
        """
        Tpos_x = self.TEmbedding(x, batch_size)
        Spos_x = self.SEmbedding(x, batch_size)
        return Tpos_x, Spos_x


class MultSpaceattention(nn.Module):
    def __init__(self, d_model, num_of_hour, d_k, d_v, n_heads, nodes, STAM):
        super(MultSpaceattention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.softmax = nn.Softmax(dim=-1)
        self.STAM = STAM
        self.num_of_hour = num_of_hour
        self.linear = nn.Sequential(
            nn.Linear(n_heads * d_v, num_of_hour),
            # nn.Dropout(p=0.1)
        )
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, sp_W_Q_params, sp_W_K_params, sp_W_V_params):
        """
        :param x: B, N, d_model
        :param sp_W_Q_params:B, N, d_model * n_heads * d_k
        :param sp_W_K_params:B, N, d_model * n_heads * d_k
        :param sp_W_V_params:B, N, d_model * n_heads * d_v
        :return:B, N, d_model
        """
        batch_size, nodes, d_model = x.shape
        w_Q = sp_W_Q_params.reshape(batch_size, nodes, d_model, self.n_heads * self.d_k)
        w_K = sp_W_K_params.reshape(batch_size, nodes, d_model, self.n_heads * self.d_k)
        w_V = sp_W_V_params.reshape(batch_size, nodes, d_model, self.n_heads * self.d_v)
        # B, H, N, d_k
        input_Q = torch.einsum('b n d,b n d h->b n h', x, w_Q).reshape(batch_size, nodes, self.n_heads,
                                                                     self.d_k).transpose(1, 2)
        input_K = torch.einsum('b n d,b n d h->b n h', x, w_K).reshape(batch_size, nodes, self.n_heads,
                                                                     self.d_k).transpose(1, 2)
        input_V = torch.einsum('b n d,b n d h->b n h', x, w_V).reshape(batch_size, nodes, self.n_heads,
                                                                     self.d_v).transpose(1, 2)

        # B, H, N, N
        A = torch.matmul(input_Q, input_K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # N, N
        STAM = torch.FloatTensor(self.STAM).cuda()
        # B, H, N, N
        B = self.softmax(A.mul(STAM))
        # B = self.dropout(B)
        # B, H, N, d_v
        output = torch.matmul(B, input_V)
        # B, N, H*d_v
        output = output.transpose(1, 2).reshape(batch_size, nodes, self.n_heads * self.d_v)
        # B, F, N, T
        output = self.linear(output).unsqueeze(1)
        return output


class GAU(nn.Module):
    def __init__(self, d_model, time_features_out, nodes):
        super(GAU, self).__init__()
        self.time_features_out = time_features_out
        self.SiLU = nn.SiLU()

        #self.gamma = nn.Parameter(torch.ones(2, time_features_out))
        #self.beta = nn.Parameter(torch.zeros(2, time_features_out))
        #nn.init.normal_(self.gamma, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(time_features_out, nodes),
            # nn.Dropout(p=0.1)
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, time_W_U_params, time_W_V_params, time_W_Z_params, time_W_qk_params):
        """
        :param x: B, T, d_model
        :param time_W_U_params:B, T, d_model*time_features_out
        :param time_W_V_params:B, T, d_model*time_features_out
        :param time_W_Z_params:B, T, d_model*time_features_out
        :param time_W_qk_params:2, time_features_out
        :return:B,F,N,T
        """
        batch_size, num_of_hour, d_model = x.shape
        time_W_V_params = time_W_V_params.reshape(batch_size, num_of_hour, d_model, self.time_features_out)
        time_W_U_params = time_W_U_params.reshape(batch_size, num_of_hour, d_model, self.time_features_out)
        time_W_Z_params = time_W_Z_params.reshape(batch_size, num_of_hour, d_model, self.time_features_out)

        v = torch.einsum('b t d, b t d o->b t o', x, time_W_V_params)  # B, T, O

        gate = torch.einsum('b t d, b t d o->b t o', x, time_W_U_params)  # B, T, O
        v = self.SiLU(v)
        gate = self.SiLU(gate)

        z = self.SiLU(torch.einsum('b t d, b t d o->b t o', x, time_W_Z_params))  # B, T, O
        time_W_qk_params = time_W_qk_params.reshape(batch_size, num_of_hour, 2, self.time_features_out)

        QK = torch.einsum('... o, ... h o -> ... h o', z, time_W_qk_params)
        #QK = torch.einsum('... o, h o -> ... h o', z, self.gamma) + self.beta  # B, T, 2, O
        q, k = QK.unbind(dim=-2)  # B, T, O
        sim = torch.einsum('b i o, b j o -> b i j', q, k) / np.sqrt(self.time_features_out)  # B, T, T

        A = self.softmax(sim)
        # A = self.dropout(A)
        V = torch.einsum('b t t, b t o-> b t o', A, v)
        V = V * gate  # B, T, O
        out = self.to_out(V)  # B, T, N
        output = out.permute(0, 2, 1).unsqueeze(1)
        return output


class STAttentionfusion(nn.Module):
    def __init__(self, num_of_features, features_out, num_of_hour):
        super(STAttentionfusion, self).__init__()
        self.local_att1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, features_out, kernel_size=1),
            nn.LayerNorm(num_of_hour),
        )

    def forward(self, GAU, SAT):
        """
        :param GAU: B, F, N, T
        :param SAT: B, F, N, T
        :return: B, F_out, N, T
        """

        ST = torch.cat([GAU, SAT], dim=1)  # B, 2F, N, T

        output = self.local_att1(ST)  # B, F_out, N, T

        return output

class create_model(nn.Module):
    def __init__(self, d_model, nodes, num_of_hour, num_of_features, d_k, d_v, n_heads, features_out,
                 time_features_out, STAM):
        super(create_model, self).__init__()
        self.Pos_Input = Pos_Input(d_model, nodes, num_of_hour, num_of_features)
        self.GAU = GAU(d_model, time_features_out, nodes)
        self.SAT = MultSpaceattention(d_model, num_of_hour, d_k, d_v, n_heads, nodes, STAM)
        self.STF = STAttentionfusion(num_of_features, features_out, num_of_hour)
        self.T_conv = nn.Conv2d(nodes, d_model, kernel_size=(1, num_of_features))
        self.S_conv = nn.Conv2d(num_of_hour, d_model, kernel_size=(1, num_of_features))
        self.x_conv = nn.Conv2d(num_of_features, features_out, kernel_size=1)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(num_of_hour)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, time_W_U_params, time_W_V_params, time_W_Z_params, time_W_qk_params, sp_W_Q_params, sp_W_K_params,
                sp_W_V_params):

        """
        :param x: B,F,N,T
        :param time_W_U_params:B, T, d_model*time_features_out
        :param time_W_V_params:B, T, d_model*time_features_out
        :param time_W_Z_params:B, T, d_model*time_features_out
        :param time_W_qk_params:2 * time_features_out
        :param sp_W_Q_params:B, N, d_model * n_heads * d_k
        :param sp_W_K_params:B, N, d_model * n_heads * d_k
        :param sp_W_V_params:B, N, d_model * n_heads * d_v
        :return: B, F_out, N, T
        """
        batch_size, num_of_features, nodes, num_of_hour = x.shape

        if num_of_features == 1:
            Tpos_x, Spos_x = self.Pos_Input(x, batch_size)  # B, F, N, T
        else:
            Tpos_x = x
            Spos_x = x

        T_x = self.T_conv(Tpos_x.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)  # B, T, d_model
        S_x = self.S_conv(Spos_x.permute(0, 3, 2, 1))[:, :, :, -1].permute(0, 2, 1)  # B, N, d_model
        T_x = self.dropout(T_x)
        S_x = self.dropout(S_x)

        GAU = self.GAU(T_x, time_W_U_params, time_W_V_params, time_W_Z_params, time_W_qk_params)  # B, F, N, T

        SAT = self.SAT(S_x, sp_W_Q_params, sp_W_K_params, sp_W_V_params)  # B, F, N, T

        STF = self.STF(GAU, SAT)

        if num_of_features == 1:
            x_conv = self.x_conv(x)
        else:
            x_conv = x

        output = self.norm(self.relu(x_conv + STF))

        return output


class run_model(nn.Module):

    def __init__(self, n_layer, d_model, nodes, num_of_hour, num_of_features, d_k, d_v,
                 n_heads, features_out, time_features_out, memory_dim, STAM):
        super(run_model, self).__init__()
        self.model_list = nn.ModuleList(
            [create_model(d_model, nodes, num_of_hour, num_of_features, d_k, d_v,
                          n_heads, features_out, time_features_out, STAM)])
        self.model_list.extend([create_model(d_model, nodes, num_of_hour, features_out, d_k, d_v,
                                             n_heads, features_out, time_features_out,
                                             STAM) for _ in range(n_layer - 1)])

        self.memory_dim = memory_dim
        self.num_of_hour = num_of_hour
        self.nodes = nodes
        self.memory_weight = self.construct_weight()

        self.time_chunk_list = []
        time_params_num = 0
        for i in range(n_layer):
            # time
            Wgc = d_model*time_features_out
            self.time_chunk_list.append(Wgc)
            time_params_num += Wgc
            Wgc = d_model*time_features_out
            self.time_chunk_list.append(Wgc)
            time_params_num += Wgc
            Wgc = d_model*time_features_out
            self.time_chunk_list.append(Wgc)
            time_params_num += Wgc
            Wgc = 2*time_features_out
            self.time_chunk_list.append(Wgc)
            time_params_num += Wgc
        self.time_fgn = nn.Sequential(nn.Linear(in_features=memory_dim, out_features=128, bias=True),
                                      nn.Linear(in_features=128, out_features=time_params_num, bias=True))

        self.space_chunk_list = []
        space_params_num = 0
        for i in range(n_layer):
            # space
            Wgc = d_model * n_heads * d_k
            self.space_chunk_list.append(Wgc)
            space_params_num += Wgc
            Wgc = d_model * n_heads * d_k
            self.space_chunk_list.append(Wgc)
            space_params_num += Wgc
            Wgc = d_model * n_heads * d_v
            self.space_chunk_list.append(Wgc)
            space_params_num += Wgc
        self.space_fgn = nn.Sequential(nn.Linear(in_features=memory_dim, out_features=128, bias=True),
                                       nn.Linear(in_features=128, out_features=space_params_num, bias=True))

        self.n_layer = n_layer
        self.conv = nn.Conv2d(num_of_hour * n_layer, 64, kernel_size=(1, features_out))
        self.Linear = nn.Linear(64, num_of_hour)

    def construct_weight(self):
        memory_weight = nn.ParameterDict()
        memory_weight['memory'] = nn.Parameter(torch.randn(128, self.memory_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['memory'])
        memory_weight['time'] = nn.Parameter(torch.randn(self.nodes, self.memory_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['time'])
        memory_weight['space'] = nn.Parameter(torch.randn(self.num_of_hour, self.memory_dim), requires_grad=True)
        nn.init.xavier_normal_(memory_weight['space'])
        return memory_weight

    def query_time_weight(self, x):
        """
        :param x: B, N, T
        """
        ht = x.permute(0, 2, 1)  # B, T, N
        query = torch.matmul(ht, self.memory_weight['time'])  # B, T, memory_dim
        att_score = torch.softmax(torch.matmul(query, self.memory_weight['memory'].t()), dim=-1)  # B, T, 64
        att_memory = torch.matmul(att_score, self.memory_weight['memory'])  # B, T, memory_dim

        params_flat = self.time_fgn(att_memory)

        params_list = torch.split(params_flat, self.time_chunk_list, dim=-1)
        time_U_params = params_list[0::4]
        time_V_params = params_list[1::4]
        time_Z_params = params_list[2::4]
        time_qk_params = params_list[3::4]
        return time_U_params, time_V_params, time_Z_params, time_qk_params

    def query_space_weight(self, x):
        """
        :param x: B, N, T
        """
        query = torch.matmul(x, self.memory_weight['space'])  # B, N, memory_dim
        att_score = torch.softmax(torch.matmul(query, self.memory_weight['memory'].t()), dim=-1)  # B, N, 64
        att_memory = torch.matmul(att_score, self.memory_weight['memory'])  # B, N, memory_dim

        params_flat = self.space_fgn(att_memory)

        params_list = torch.split(params_flat, self.space_chunk_list, dim=-1)
        space_Q_params = params_list[0::3]
        space_K_params = params_list[1::3]
        space_V_params = params_list[2::3]
        return space_Q_params, space_K_params, space_V_params

    def forward(self, x):
        output = []
        time_U_params, time_V_params, time_Z_params, time_qk_params = self.query_time_weight(x)
        space_Q_params, space_K_params, space_V_params = self.query_space_weight(x)
        x = x.unsqueeze(1)
        for i in range(self.n_layer):
            time_W_U_params, time_W_V_params, time_W_Z_params, time_W_qk_params = \
            time_U_params[i], time_V_params[i], time_Z_params[i], time_qk_params[i]
            sp_W_Q_params, sp_W_K_params, sp_W_V_params = space_Q_params[i], space_K_params[i], space_V_params[i]
            x = self.model_list[i](x, time_W_U_params, time_W_V_params, time_W_Z_params, time_W_qk_params, sp_W_Q_params,
                                   sp_W_K_params, sp_W_V_params)
            output.append(x)

        output = torch.cat(output, dim=-1)
        output = self.conv(output.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        output = self.Linear(output).squeeze(1)

        return output
