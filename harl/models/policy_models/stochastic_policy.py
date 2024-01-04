import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space



class HetGAT(nn.Module):
    '''
        HOA-Net
    '''
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_type=3, n_heads=1, add_self=True, add_elu=True):
        super(HetGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_type = num_type
        self.n_heads = n_heads
        self.add_self = add_self
        self.add_elu = add_elu
        self.negative_slope = 0.2

        assert n_heads == 1, '目前还没有涉及多头的形式！'
        assert self.add_self, ('Het-GAT need to add self info!')

        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim, hidden_dim)) for _ in range(self.num_type)])
        self.a = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_dim * 2, 1)) for _ in range(self.num_type)])
        self.act = nn.LeakyReLU(negative_slope=self.negative_slope)

        # =========Xvaier init=========
        # for i in range(self.num_type):
        #     nn.init.xavier_uniform_(self.W[i].data, gain=1.414)
        #     nn.init.xavier_uniform_(self.a[i].data, gain=1.414)
        # =========Kaiming init=========
        for i in range(self.num_type):
            nn.init.kaiming_uniform_(self.W[i].data, a=self.negative_slope)  # a 是 Leaky ReLU 的负斜率
            nn.init.kaiming_uniform_(self.a[i].data, a=self.negative_slope)

    
    def forward(self, h, num_ally, num_opp, ALL_TYPE_MASK):
        assert self.num_type == ALL_TYPE_MASK.shape[0], ('Wrong mask!') # ALL_TYPE_MASK的维度（type, batch, agent) 
        assert 1 + num_ally + num_opp == ALL_TYPE_MASK.shape[-1], ('Wrong mask!')
        assert 1 + num_ally + num_opp == h.shape[-2]

        # h suppose to be the shape of [B, 1+num_ally+num_opp, C]
        device = h.device

        # construct mask    ALL_TYPE_MASK: (T, B, N) = mask_self (T, B, 1) + mask_other (T, B, N-1)
        (mask_self, mask_ally, mask_opp)= (ALL_TYPE_MASK[:,:,0:1], ALL_TYPE_MASK[:,:, 1:1+num_ally], ALL_TYPE_MASK[:,:, 1+num_ally:])   
        (h_self, h_ally, h_opp) = (h[:, 0:1, :], h[:, 1:1+num_ally, :], h[:, 1+num_ally:, :]) 
        T, B, _= mask_ally.size() # NN = N-1
        # (T, B, N-1) --> (T, T, B, N-1) --> (B, T, T, N-1) --> (B, T*T*N-1)
        MASK_ALLY = mask_ally.unsqueeze(0).repeat(T, 1,1,1).permute(2,0,1,3).reshape(B, T*T*num_ally)
        MASK_OPP  = mask_opp.unsqueeze(0).repeat(T, 1,1,1).permute(2,0,1,3).reshape(B, T*T*num_opp)

        # construct info
        H_SELF = torch.zeros(self.num_type, h_self.shape[0], h_self.shape[1], self.hidden_dim, device=device) # (T, B, 1, C)
        H_ALLY = torch.zeros(self.num_type, h_ally.shape[0], h_ally.shape[1], self.hidden_dim, device=device) # (T, B, N-1, C)
        H_OPP = torch.zeros(self.num_type, h_opp.shape[0], h_opp.shape[1], self.hidden_dim, device=device)    # (T, B, N-1, C)

        # Linear Transform
        for i in range(self.num_type):
            H_SELF[i,...] = torch.matmul(h_self, self.W[i])
            H_ALLY[i,...] = torch.matmul(h_ally, self.W[i])
            H_OPP[i,...]  = torch.matmul(h_opp, self.W[i])

        # H_SELF (T, B, 1, C) --> (T, [T], B, 1, C) --> (T, T, B, N-1, C)
        copy_times = max(num_ally, num_opp)
        H_SELF_INP = H_SELF.unsqueeze(1).repeat(1,T,1,copy_times,1)

        # calculate ally embedding
        T, B, NN, C = H_ALLY.size() 
        # H_ALLY (T, B, N-1, C) --> ([T], T, B, N-1, C)
        H_ALLY_INP = H_ALLY.unsqueeze(0).repeat(T, 1,1,1,1)
        ALLY_INPUT = torch.cat([H_SELF_INP[...,:num_ally,:],H_ALLY_INP], dim=-1) # (T, T, B, N-1, 2*C)
        E_ALLY = torch.zeros(self.num_type, self.num_type, B, num_ally, 1, device=device) # (T, T, B, N-1, 1)
        for i in range(self.num_type):
            E_ALLY[:,i,:,:,:] = torch.matmul(ALLY_INPUT[:,i,:,:,:], self.a[i])
        # E_ALLY (T, T, B, N-1, 1) --> (B, T, T, N-1, 1) --> (B, T*T*N-1)
        E_ALLY_squeeze = E_ALLY.permute(2, 0, 1, 3, 4).reshape(B, T*T*num_ally*1)
        E_ALLY_squeeze[MASK_ALLY] = float(-1e6)
        ally_attention = F.softmax(E_ALLY_squeeze, dim=-1) # (B, T*T*N-1)

        #防止全是inf的情况算出1/N的权重来
        ally_attnc = ally_attention.clone()
        ally_attnc[MASK_ALLY] = 0
        ally_attention = ally_attnc

        # (T, T, B, N-1, C) --> (B, T*T*N-1, C)
        H_ALLY_INP = H_ALLY_INP.permute(2,0,1,3,4).reshape(B, T*T*num_ally, C)
        # final matmul: (B, 1, T*T*N-1) * (B, T*T*N-1, C) = (B, 1, C) --> (B, C)
        ally_weighted_sum = torch.matmul(ally_attention.unsqueeze(1), H_ALLY_INP).squeeze(1)
        
        # calculate opp embedding
        T, B, NN, C = H_OPP.size() # NN = N-1
        # H_OPP (T, B, N-1, C) --> ([T], T, B, N-1, C)
        H_OPP_INP = H_OPP.unsqueeze(0).repeat(T, 1,1,1,1)
        OPP_INPUT = torch.cat([H_SELF_INP[...,:num_opp,:],H_OPP_INP], dim=-1) # (T, T, B, N-1, 2*C)
        E_OPP = torch.zeros(self.num_type, self.num_type, B, num_opp, 1, device=device) # (T, T, B, N-1, 1)
        for i in range(self.num_type):
            E_OPP[:,i,:,:,:] = torch.matmul(OPP_INPUT[:,i,:,:,:], self.a[i])
        # E_OPP (T, T, B, N-1, 1) --> (B, T, T, N-1, 1) --> (B, T*T*N-1)
        E_OPP_squeeze = E_OPP.permute(2, 0, 1, 3, 4).reshape(B, T*T*num_opp*1)
        E_OPP_squeeze[MASK_OPP] = float(-1e6)
        opp_attention = F.softmax(E_OPP_squeeze, dim=-1) # (B, T*T*N-1)

        #防止全是inf的情况算出1/N的权重来
        opp_attnc = opp_attention.clone()
        opp_attnc[MASK_OPP] = 0
        opp_attention = opp_attnc

        # (T, T, B, N-1, C) --> (B, T*T*N-1, C)
        H_OPP_INP = H_OPP_INP.permute(2,0,1,3,4).reshape(B, T*T*num_opp, C)
        # final matmul: (B, 1, T*T*N-1) * (B, T*T*N-1, C) = (B, 1, C) --> (B, C)
        opp_weighted_sum = torch.matmul(opp_attention.unsqueeze(1), H_OPP_INP).squeeze(1)

        # mask_self (T, B, 1) H_SELF (T, B, 1, C)
        mask_self_expand = mask_self.unsqueeze(-1).repeat(1,1,1,H_SELF.shape[-1])
        h_self_sum = (H_SELF * mask_self_expand).sum(dim=0).squeeze()

        if self.add_elu:
            H_E = F.elu(h_self_sum + ally_weighted_sum + opp_weighted_sum) 
        else:
            H_E = (h_self_sum + ally_weighted_sum + opp_weighted_sum) 

        return H_E.view(B, self.hidden_dim)




class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), special_NN=None):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.special_NN = special_NN

        # NOTE 在这里提取JSON中的地图信息，然后根据地图信息来确定输入的维度
        if self.special_NN in ["HOA-Net"]:
            # extract map info
            import json
            file_path = '/home/hutianyi/HARL-HetGNN/harl/envs/smac/prior_knowledge.json'
            with open(file_path, "r") as file:
                self.prior_knowledge = json.load(file)
                self.num_ally  = self.prior_knowledge['ally_feats_dim'][0]
                self.num_opp = self.prior_knowledge['enemy_feats_dim'][0]
                self.num_type  = self.prior_knowledge["num_type"]
                self.own_feats_dim = self.prior_knowledge["own_feats_dim"]
                self.move_feats_dim = self.prior_knowledge["move_feats_dim"]
                self.num_actions = self.own_feats_dim - self.prior_knowledge['enemy_feats_dim'][-1]
                self.num_agents = self.num_ally + 1
                assert self.num_actions == action_space.n, ('Wrong num_actions!')

            # construct networks
            # TODO 这里没有加观测预处理网络，之后要加上看看效果如何（直接将HetGAT当做base
            preprocess = MLPBase
            self.preprocess_layer = preprocess(args, [self.own_feats_dim])
            self.base = HetGAT(self.hidden_sizes[-1], self.hidden_sizes[-1], output_dim = self.hidden_sizes[-1], num_type = self.num_type)
            # self.base = HetGAT(self.own_feats_dim, self.hidden_sizes[-1], output_dim = self.hidden_sizes[-1], num_type = self.num_type)
            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                self.rnn = RNNLayer(
                    self.hidden_sizes[-1] + self.move_feats_dim + self.num_agents,
                    self.hidden_sizes[-1] + self.move_feats_dim + self.num_agents,
                    self.recurrent_n,
                    self.initialization_method,
                )
            self.act = ACTLayer(
                action_space,
                self.hidden_sizes[-1] + self.move_feats_dim + self.num_agents,
                self.initialization_method,
                self.gain,
                args,
            )

        else:
            obs_shape =  (obs_space)
            base = CNNBase if len(obs_shape) == 3 else MLPBase            
            self.base = base(args, obs_shape)

            if self.use_naive_recurrent_policy or self.use_recurrent_policy:
                self.rnn = RNNLayer(
                    self.hidden_sizes[-1],
                    self.hidden_sizes[-1],
                    self.recurrent_n,
                    self.initialization_method,
                )

            self.act = ACTLayer(
                action_space,
                self.hidden_sizes[-1],
                self.initialization_method,
                self.gain,
                args,
            )

        self.to(device)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        if self.special_NN in ["HOA-Net"]:
            # ============== NOTE ==============
            # Below are the new things
            avai_obs, HetGraph_obs = self.reconstruct_observation(obs)
            HetGraph_mask = self.calculate_mask(HetGraph_obs)
            Preprocessed_obs = self.preprocess_layer(HetGraph_obs)
            aggeregated_obs = self.base(Preprocessed_obs, self.num_ally, self.num_opp, HetGraph_mask)
            actor_features = torch.cat([avai_obs, aggeregated_obs], dim=-1)
            # ============== NOTE ==============
        else:
            # avai_obs, HetGraph_obs = self.reconstruct_observation(obs)
            # B, N, C = HetGraph_obs.shape
            # HetGraph_obs = HetGraph_obs.view(B,N*C)
            # obs = torch.cat([avai_obs, HetGraph_obs], dim=-1)
            actor_features = self.base(obs)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        if self.special_NN in ["HOA-Net"]:
            # ============== NOTE ==============
            # Below are the new things
            avai_obs, HetGraph_obs = self.reconstruct_observation(obs)
            HetGraph_mask = self.calculate_mask(HetGraph_obs)
            Preprocessed_obs = self.preprocess_layer(HetGraph_obs)
            aggeregated_obs = self.base(Preprocessed_obs, self.num_ally, self.num_opp, HetGraph_mask)
            actor_features = torch.cat([avai_obs, aggeregated_obs], dim=-1)
            # ============== NOTE ==============
        else:
            # avai_obs, HetGraph_obs = self.reconstruct_observation(obs)
            # B, N, C = HetGraph_obs.shape
            # HetGraph_obs = HetGraph_obs.view(B,N*C)
            # obs = torch.cat([avai_obs, HetGraph_obs], dim=-1)
            actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy, action_distribution

    def reconstruct_observation(self, obs):
        '''
        This is only for SMAC envs.
        Assert Have agent_id included in obs.
        input:  obs [B, complex_dim]
        return: avai_obs [B, move_feats_dim] + shaped_obs [B, self + num_ally + num_opp, own_feats_dim]
        '''
        
        B = obs.shape[0]

        X = self.prior_knowledge['enemy_feats_dim'][-1]
        assert self.num_actions + X == self.own_feats_dim, ('Wrong own_feats_dim or X compute!')

        temp_opp_dim = self.num_opp * X
        temp_ally_dim = self.num_ally * (X + self.num_actions)
        temp_self_dim = 1 * (X + self.num_actions)

        temp_ally = obs[:, :temp_ally_dim]                                                    # [B, num_ally * (X+num_actions)]
        temp_opp  = obs[:, temp_ally_dim:temp_ally_dim + temp_opp_dim]                        # [B, num_opp * X]
        temp_self = obs[:, temp_ally_dim + temp_opp_dim + self.move_feats_dim:
                        temp_ally_dim + temp_opp_dim + self.move_feats_dim + temp_self_dim]   # [B, 1 * (X+num_actions)]
        temp_move_feats = obs[:, temp_ally_dim + temp_opp_dim:temp_ally_dim + temp_opp_dim + self.move_feats_dim]
        temp_agent_id   = obs[:, temp_ally_dim + temp_opp_dim + self.move_feats_dim + temp_self_dim:]

        temp_opp = temp_opp.view(B, self.num_opp, X)
        temp_ally = temp_ally.view(B, self.num_ally, X + self.num_actions)
        temp_self = temp_self.view(B, 1, X + self.num_actions)

        avai_obs = torch.cat([temp_move_feats, temp_agent_id], dim=-1)

        temp_opp = torch.cat([temp_opp, torch.zeros(B, self.num_opp, self.num_actions).to(obs.device)], dim=2)

        shaped_obs = torch.cat([temp_self, temp_ally, temp_opp],dim=1)

        return avai_obs, shaped_obs
    
    def calculate_mask(self, HetGraph_obs):
        '''
        Only for SMAC envs.
        input:  HetGraph_obs [B, self + num_ally + num_opp, own_feats_dim]
        output: ALL_TYPE_MASK [num_type, B, self + num_ally + num_opp]
        '''
        # Remove the last 'num_actions' dimensions
        HetGraph_obs_trimmed = HetGraph_obs[:, :, :-self.num_actions]

        # Reshape to [own_feats_dim-num_actions, B, self + num_ally + num_opp]
        HetGraph_obs_reshaped = HetGraph_obs_trimmed.permute(2, 0, 1)

        # Take only the last 'num_type' dimensions from the first axis
        ALL_TYPE_MASK = HetGraph_obs_reshaped[-self.num_type:, :, :].bool()

        return ALL_TYPE_MASK


