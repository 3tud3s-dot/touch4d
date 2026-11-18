import torch


def mean_std_attn_dense_rotation_unify(query, receiver_emb, sender_emb, inter_sender_emb, self_attn_mask, inter_attn_mask, scale, eps=1e-7, **kwargs):
    '''
        attn_mask: bs, N_r, num_neighbor
        query: bs, num_head/mask_type, mask_type(1), N_r, dim
        receiver_emb: bs, num_head/mask_type, mask_type(1), N_r, dim
        sender_emb: bs, num_head/mask_type, mask_type(1), N_r, num_neighbor, dim
    '''
    bs, _num_head, _mask_type, q_N, head_dim = query.shape
    N_r = receiver_emb.shape[3]
    N_s = sender_emb.shape[3]
    N_nn = inter_sender_emb.shape[4]

    self_attn_mask = self_attn_mask.unsqueeze(1)
    inter_attn_mask = inter_attn_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
    # u = (r+s)/2
    # q(r+s-u)/e = q(r+s)/(2e)
    # e^2 = [(r-u)^T(r-u) + (s-u)^T(s-u)]/2 = (r^2+s^2-2rs)/4

    # Calculate self attention weight
    ## bs, _num_head, _mask_type, q_N, 1
    r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    ## bs, _num_head, _mask_type, 1, q_N
    s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    # N_r, N_s
    rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
    self_var_rs = (r_square + s_square-2*rs)/4
    # N_r, N_s
    self_std_rs = torch.sqrt(self_var_rs)
    # r = a_w * ((r-u)/e) + a_b, before attention
    ## This is a must: in case the embedding is inplacely changed
    self_attn = torch.matmul(query, sender_emb.transpose(-1, -2)) + torch.sum(query * receiver_emb, dim=-1, keepdim=True)
    self_attn /= (self_std_rs+eps)
    self_attn = self_attn * scale
    self_attn.masked_fill_(~self_attn_mask, float('-inf'))
    # self_attn = torch.exp(self_attn)

    # Calculate inter attention weight
    ## Inter will omit the mesh connection
    ### mask_type 0: world mask; 1: mesh mask
    inter_query = query[:, :, 0:1].unsqueeze(4)
    inter_receiver = receiver_emb[:, :, 0:1].unsqueeze(4)
    inter_sender = inter_sender_emb[:, :, 0:1]
    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    # e^2 = [(r-u)^T(r-u) + (s-u)^T(s-u)]/2 = (r^2+s^2-2rs)/4
    inter_var_rs = torch.sum((inter_receiver - inter_sender)**2, dim=-1, keepdim=True) / 4
    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    inter_std_rs = torch.sqrt(inter_var_rs)
    # r = a_w * ((r-u)/e) + a_b, before attention
    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    raw_inter_attn = torch.sum(inter_query*inter_sender, dim=-1, keepdim=True) + torch.sum(inter_query*inter_receiver, dim=-1, keepdim=True)
    inter_attn =raw_inter_attn / (inter_std_rs+eps)
    inter_attn = inter_attn * scale
    inter_attn.masked_fill_(~inter_attn_mask, float('-inf'))
    # bs, num_head/mask_type, _, N_r, num_neighbor
    inter_attn = inter_attn.squeeze(-1)
    # assert torch.where(torch.isnan(inter_attn))[0].size(0) == 0
    # inter_attn = torch.exp(inter_attn)
    # assert torch.where(torch.isnan(inter_attn))[0].size(0) == 0
    
    # Joint softmax
    ## bs, num_head/mask_type, _, N_r, N_s+num_neighbor
    ### mask_type 0: world mask; 1: mesh mask
    joint_attn = torch.cat([
        self_attn,
        torch.cat([inter_attn, torch.full_like(inter_attn, float('-inf')).to(inter_attn)], dim=2)
    ], dim=-1)
    joint_attn = joint_attn.softmax(dim=-1)

    # Split attn
    self_attn_normed = joint_attn[..., :N_s]
    inter_attn_normed = joint_attn[:, :, 0:1, :, N_s:].unsqueeze(-1)

    # Clean NaN
    self_attn_normed = self_attn_normed.masked_fill(~self_attn_mask, 0.0).clamp(-5, 5)
    inter_attn_normed = inter_attn_normed.masked_fill(~inter_attn_mask, 0.0).clamp(-5, 5)
    assert torch.where(torch.isnan(self_attn_normed))[0].size(0) == 0
    assert torch.where(torch.isnan(inter_attn_normed))[0].size(0) == 0

    # ## bs, head/mesh_type, mesh_type, N_r
    # Z_self_attn = torch.sum(self_attn, dim=-1)
    # ## bs, head/mesh_type, 1, N_r
    # Z_inter_attn = torch.sum(inter_attn, dim=(-1, -2))
    
    # ## bs, head/mesh_type, 1, N_r
    # shared_Z = Z_self_attn[:, :, 0:1] + Z_inter_attn
    # assert torch.where(torch.isnan(shared_Z))[0].size(0) == 0

    # # Normalize both attention, softmax
    # ## Mesh part
    # # To omit inplace modification
    # self_attn_normed_world = self_attn[:, :, 0:1] / (shared_Z.unsqueeze(-1) + eps)
    # self_attn_normed_mesh = self_attn[:, :, 1:] / (Z_self_attn[:, :, 1:].unsqueeze(-1) + eps)
    # self_attn_normed = torch.cat([self_attn_normed_world, self_attn_normed_mesh], dim=2)
    # # ## World part
    # inter_attn_normed = inter_attn / (shared_Z.unsqueeze(-1).unsqueeze(-1) + eps)
    # assert torch.where(torch.isnan(shared_Z))[0].size(0) == 0
    # assert torch.where(torch.isnan(inter_attn))[0].size(0) == 0
    # # Clear NaN
    # self_attn_normed = self_attn_normed.masked_fill(~self_attn_mask, 0.0).clamp(-5, 5)
    # inter_attn_normed = inter_attn_normed.masked_fill(~inter_attn_mask, 0.0)
    # inter_attn_normed = inter_attn_normed.clamp(-5, 5)
    # assert torch.where(torch.isnan(self_attn_normed[:, :, 0:1]))[0].size(0) == 0
    # assert torch.where(torch.isnan(self_attn_normed[:, :, 1:]))[0].size(0) == 0
    # # assert torch.where(torch.isnan(inter_attn_normed))[0].size(0) == 0
    # if torch.where(torch.isnan(inter_attn_normed))[0].size(0) != 0:
    #     nan_loc = torch.where(torch.isnan(inter_attn_normed))
    #     nan_size = nan_loc[0].size(0)
    #     zero_size = torch.where(shared_Z == 0)[0].size(0)
    #     epszero_size = torch.where((shared_Z.unsqueeze(-1).unsqueeze(-1) + eps) == 0)[0].size(0)
    #     print(f"\nNaN: {torch.where(torch.isnan(inter_attn_normed))}\n==>nan size {nan_size}\n==>Zero size {zero_size}\n==>epszero {epszero_size}\n==>corresponding value in\n==>attn_normed {inter_attn_normed[nan_loc]}\n==>original {inter_attn[nan_loc]}\n")
    #     print(f"==> Std: {inter_std_rs[nan_loc]}\n==>Raw attn: {raw_inter_attn[nan_loc]}\n")
    #     print(f"==> Normalizer: {Z_inter_attn[nan_loc[:4]]}\n==> Shared normalizer: {shared_Z[nan_loc[0:4]]}")
    #     assert False
    
    # For aggregation
    ## w * ((r-u)/e) = w * (r-s)/(2e) = w/(2e) * (r-s)
    self_attn_normed = self_attn_normed / (self_std_rs * 2)
    inter_attn_normed = inter_attn_normed / (inter_std_rs * 2)

    # bs, num_heads / mask_type, mask_type, N_r, N_s
    # bs, num_heads / mask_type, 1, N_r, K, 1
    return self_attn_normed, inter_attn_normed


def mean_std_attn_dense_rotation(query, receiver_emb, sender_emb, attn_mask, scale, eps=1e-7, **kwargs):
    # emb: bs, num_head, N, head_dim
    attn_mask = attn_mask.unsqueeze(1)
    bs, _num_head, _mask_type, q_N, head_dim = query.shape
    r_N = receiver_emb.shape[3]

    # u = (r+s)/2
    # q(r+s-u)/e = q(r+s)/(2e)
    # e^2 = [(r-u)^T(r-u) + (s-u)^T(s-u)]/2 = (r^2+s^2-2rs)/4
    r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    # N_r, N_s
    rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
    # var_rs = rs + r_square + s_square + mean_rs*mean_rs - (2/dim) * mean_rs*(mean_receiver + mean_sender)
    var_rs = (r_square + s_square-2*rs)/4
    # N_r, N_s
    std_rs = torch.sqrt(var_rs)

    # r = a_w * ((r-u)/e) + a_b, before attention
    ## This is a must: in case the embedding is inplacely changed
    attn = torch.matmul(query, sender_emb.transpose(-1, -2)) + torch.sum(query * receiver_emb, dim=-1, keepdim=True)
    attn /= std_rs

    attn = attn * scale
    # attn = self.attn_activate(attn)
    # qs_attn = torch.matmul(q, v_sender.transpose(-2, -1)) * self.scale
    # # B, num_heads, N, 1
    # qr_attn = (q * v_receiver).sum(dim=-1).unsqueeze(-1) * self.scale
    # mask = (qs_attn - qr_attn) < 0
    # attn = qs_attn.masked_fill(mask, float('-inf'))
    # TODO: The attn_mask can have identity for n_shapes, thus there'll be no nan after softmax
    attn.masked_fill_(~attn_mask, float('-inf'))

    # attn B, num_heads, N, N
    attn = attn.softmax(dim=-1)
    attn = attn.masked_fill(~attn_mask, 0.0)
    assert torch.where(torch.isnan(attn))[0].size(0) == 0
    attn = attn.clamp(-5, 5)

    attn_std = attn / std_rs
    attn_std /= 2 # this is due to the mean

    # bs, num_heads / mask_type, mask_type, N, head_dim
    return attn_std


def mean_std_attn_dense_rotation_obstacle(query, receiver_emb, sender_emb, attn_mask, scale, eps=1e-7, **kwargs):
    '''
        attn_mask: bs, N_r, num_neighbor
        query: bs, num_head/mask_type, mask_type(1), N_r, dim
        receiver_emb: bs, num_head/mask_type, mask_type(1), N_r, dim
        sender_emb: bs, num_head/mask_type, mask_type(1), N_r, num_neighbor, dim
    '''
    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
    bs, _num_head, _mask_type, q_N, head_dim = query.shape
    r_N = receiver_emb.shape[3]
    query = query.unsqueeze(4)
    receiver_emb = receiver_emb.unsqueeze(4)

    # u = (r+s)/2
    # q(r+s-u)/e = q(r+s)/(2e)
    # e^2 = [(r-u)^T(r-u) + (s-u)^T(s-u)]/2 = (r^2+s^2-2rs)/4
    # bs, num_head/mask_type, mask_type, N_r, num_neighbor, dim
    mean_rs = (receiver_emb + sender_emb) / 2
    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    var_rs = (torch.sum((receiver_emb - mean_rs)**2, dim=-1, keepdim=True) \
        + torch.sum((sender_emb - mean_rs)**2, dim=-1, keepdim=True)) / 2
    
    # r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    # s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True)
    # # N_r, N_s
    # rs = torch.sum(receiver_emb * sender_emb, dim=-1, keepdim=True)
    # # var_rs = rs + r_square + s_square + mean_rs*mean_rs - (2/dim) * mean_rs*(mean_receiver + mean_sender)
    # var_rs = (r_square + s_square-2*rs)/4

    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    std_rs = torch.sqrt(var_rs)

    # r = a_w * ((r-u)/e) + a_b, before attention
    # bs, num_head/mask_type, _, N_r, num_neighbor, 1
    attn = torch.sum(query*sender_emb, dim=-1, keepdim=True) + torch.sum(query*receiver_emb, dim=-1, keepdim=True)
    attn /= std_rs

    attn = attn * scale
    attn.masked_fill_(~attn_mask, float('-inf'))
    # attn B, num_heads, N, N
    attn = attn.squeeze(-1).softmax(dim=-1).unsqueeze(-1)
    attn = attn.masked_fill(~attn_mask, 0.0)
    assert torch.where(torch.isnan(attn))[0].size(0) == 0
    attn = attn.clamp(-5, 5)

    attn_std = attn / std_rs
    attn_std /= 2 # this is due to the mean

    # bs, num_heads / mask_type, mask_type, N, head_dim, 1
    return attn_std

def mean_std_attn_dense(query, receiver_emb, sender_emb, attn_mask, scale, eps=1e-7, **kwargs):
    # emb: bs, num_head, N, head_dim
    mask_type = attn_mask.shape[1]
    attn_mask = attn_mask.unsqueeze(1)
    assert receiver_emb.shape[1] % mask_type == 0
    bs, num_head, q_N, head_dim = query.shape
    r_N = receiver_emb.shape[2]
    s_N = sender_emb.shape[2]

    query = query.reshape(bs, num_head // mask_type, mask_type, q_N, head_dim)
    receiver_emb = receiver_emb.reshape(bs, num_head // mask_type, mask_type, r_N, head_dim)
    sender_emb = sender_emb.reshape(bs, num_head // mask_type, mask_type, s_N, head_dim)
    dim = receiver_emb.shape[-1]
    # attn_mask = attn_mask.unsqueeze(1)

    mean_receiver = torch.mean(receiver_emb, dim=-1, keepdim=True)
    mean_sender = torch.mean(sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    # N_r, N_s
    mean_rs = mean_receiver + mean_sender

    r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
    s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
    # N_r, N_s
    rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
    # var_rs = rs + r_square + s_square + mean_rs*mean_rs - (2/dim) * mean_rs*(mean_receiver + mean_sender)
    var_rs = (2*rs + r_square + s_square)/dim  - mean_rs*mean_rs + eps
    # N_r, N_s
    std_rs = torch.sqrt(var_rs)

    # mean_rs = mean_rs * (attn_mask)
    # std_rs = std_rs.masked_fill(~attn_mask, 1.0)

    # Original attn
    ## bs, N_r, dim; bs, N_s, dim -> bs, N_r, N_s
    # assert torch.where(torch.isnan(query))[0].size(0) == 0, f"query: {query[torch.where(torch.isnan(query))]} with index: {torch.where(torch.isnan(query))}"
    # assert torch.where(torch.isnan(sender_emb))[0].size(0) == 0, f"query: {sender_emb[torch.where(torch.isnan(sender_emb))]} with index: {torch.where(torch.isnan(sender_emb))}"
    # assert torch.where(torch.isnan(receiver_emb))[0].size(0) == 0, f"query: {receiver_emb[torch.where(torch.isnan(receiver_emb))]} with index: {torch.where(torch.isnan(receiver_emb))}"

    # r = a_w * ((r-u)/e) + a_b, before attention
    decentral_receiver_emb = receiver_emb - mean_receiver
    decentral_sender_emb = sender_emb - mean_sender.transpose(-1, -2)
    attn = torch.matmul(query, decentral_sender_emb.transpose(-1, -2)) + torch.sum(query * decentral_receiver_emb, dim=-1, keepdim=True)
    attn /= std_rs

    attn = attn * scale
    # attn = self.attn_activate(attn)
    # qs_attn = torch.matmul(q, v_sender.transpose(-2, -1)) * self.scale
    # # B, num_heads, N, 1
    # qr_attn = (q * v_receiver).sum(dim=-1).unsqueeze(-1) * self.scale
    # mask = (qs_attn - qr_attn) < 0
    # attn = qs_attn.masked_fill(mask, float('-inf'))
    # TODO: The attn_mask can have identity for n_shapes, thus there'll be no nan after softmax
    attn.masked_fill_(~attn_mask, float('-inf'))

    # attn B, num_heads, N, N
    attn = attn.softmax(dim=-1)
    attn = attn.masked_fill(~attn_mask, 0.0)
    # assert torch.where(torch.isnan(attn))[0].size(0) == 0
    attn = attn.clamp(-5, 5)

    attn_std = attn / std_rs
    # mean_rs = mean_rs * attn_std
    # x = (torch.matmul(attn_std, sender_emb) - torch.sum(mean_rs, dim=-1, keepdim=True))

    x = torch.matmul(attn_std, decentral_sender_emb)
    x = x + torch.sum(attn_std, dim=-1, keepdim=True) * decentral_receiver_emb
    # bs, num_heads/type, type, N, head_dim
    x = x.reshape(bs, num_head, r_N, head_dim)

    # bs, num_heads, N, head_dim
    return x, attn

# This one has bug
# def mean_std_attn_dense(query, receiver_emb, sender_emb, attn_mask, scale, eps=1e-7):
#     # emb: bs, num_head, N, head_dim
#     mask_type = attn_mask.shape[1]
#     attn_mask = attn_mask.unsqueeze(1)
#     assert receiver_emb.shape[1] % mask_type == 0
#     bs, num_head, q_N, head_dim = query.shape
#     r_N = receiver_emb.shape[2]
#     s_N = sender_emb.shape[2]

#     query = query.reshape(bs, num_head // mask_type, mask_type, q_N, head_dim)
#     receiver_emb = receiver_emb.reshape(bs, num_head // mask_type, mask_type, r_N, head_dim)
#     sender_emb = sender_emb.reshape(bs, num_head // mask_type, mask_type, s_N, head_dim)
#     dim = receiver_emb.shape[-1]
#     # attn_mask = attn_mask.unsqueeze(1)

#     mean_receiver = torch.mean(receiver_emb, dim=-1, keepdim=True)
#     mean_sender = torch.mean(sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
#     # N_r, N_s
#     mean_rs = mean_receiver + mean_sender

#     r_square = torch.sum(receiver_emb * receiver_emb, dim=-1, keepdim=True)
#     s_square = torch.sum(sender_emb * sender_emb, dim=-1, keepdim=True).transpose(-1, -2)
#     # N_r, N_s
#     rs = torch.matmul(receiver_emb, sender_emb.transpose(-1, -2))
#     # var_rs = rs + r_square + s_square + mean_rs*mean_rs - (2/dim) * mean_rs*(mean_receiver + mean_sender)
#     var_rs = (2*rs + r_square + s_square)/dim  - mean_rs*mean_rs + eps
#     # N_r, N_s
#     std_rs = torch.sqrt(var_rs)

#     # mean_rs = mean_rs * (attn_mask)
#     # std_rs = std_rs.masked_fill(~attn_mask, 1.0)

#     # Original attn
#     ## bs, N_r, dim; bs, N_s, dim -> bs, N_r, N_s
#     # assert torch.where(torch.isnan(query))[0].size(0) == 0, f"query: {query[torch.where(torch.isnan(query))]} with index: {torch.where(torch.isnan(query))}"
#     # assert torch.where(torch.isnan(sender_emb))[0].size(0) == 0, f"query: {sender_emb[torch.where(torch.isnan(sender_emb))]} with index: {torch.where(torch.isnan(sender_emb))}"
#     # assert torch.where(torch.isnan(receiver_emb))[0].size(0) == 0, f"query: {receiver_emb[torch.where(torch.isnan(receiver_emb))]} with index: {torch.where(torch.isnan(receiver_emb))}"
#     attn = torch.matmul(query, sender_emb.transpose(-1, -2)) + torch.sum(query * receiver_emb, dim=-1, keepdim=True)

#     attn = attn * scale
#     # attn = self.attn_activate(attn)
#     # qs_attn = torch.matmul(q, v_sender.transpose(-2, -1)) * self.scale
#     # # B, num_heads, N, 1
#     # qr_attn = (q * v_receiver).sum(dim=-1).unsqueeze(-1) * self.scale
#     # mask = (qs_attn - qr_attn) < 0
#     # attn = qs_attn.masked_fill(mask, float('-inf'))
#     # TODO: The attn_mask can have identity for n_shapes, thus there'll be no nan after softmax
#     attn.masked_fill_(~attn_mask, float('-inf'))

#     # attn B, num_heads, N, N
#     attn = attn.softmax(dim=-1)
#     attn = attn.masked_fill(~attn_mask, 0.0)
#     # assert torch.where(torch.isnan(attn))[0].size(0) == 0
#     attn = attn.clamp(-5, 5)

#     attn_std = attn / std_rs
#     # mean_rs = mean_rs * attn_std
#     # x = (torch.matmul(attn_std, sender_emb) - torch.sum(mean_rs, dim=-1, keepdim=True))

#     x = torch.matmul(attn_std, sender_emb) - torch.sum(mean_rs*attn_std, dim=-1, keepdim=True)
#     x = x + torch.sum(attn_std, dim=-1, keepdim=True) * receiver_emb
#     # bs, num_heads/type, type, N, head_dim
#     x = x.reshape(bs, num_head, r_N, head_dim)

#     # bs, num_heads, N, head_dim
#     return x, attn


def mean_std_attn(query, receiver_emb, sender_emb, r_mask, s_mask, scale, eps=1e-7, attn_weight=None, **kwargs):
    '''
        r_mask: bs*head, N_r, N_e
    '''
    # emb: bs, num_head, N, head_dim
    bs, num_head, N_r, head_dim = receiver_emb.shape
    bs, num_head, N_s, head_dim = sender_emb.shape

    receiver_emb = receiver_emb.reshape(-1, N_r, head_dim)
    sender_emb = sender_emb.reshape(-1, N_s, head_dim)
    query = query.reshape(-1, N_r, head_dim)

    # bs*head, N_r, N_e, bs*head, N_r, dim --> bs*head, N_e, dim
    e_r_emb = torch.bmm(r_mask.transpose(-1, -2), receiver_emb)
    e_s_emb = torch.bmm(s_mask.transpose(-1, -2), sender_emb)
    e_emb = e_r_emb + e_s_emb
    mean_rs = torch.mean(e_emb, dim=-1, keepdim=True)
    std_rs = torch.std(e_emb, dim=-1, keepdim=True)
    e_emb = (e_emb - mean_rs) / (std_rs + eps)

    e_q_emb = torch.bmm(r_mask.transpose(-1, -2), query)
    # bs*head, N_e, 1
    attn = torch.sum(e_q_emb * e_emb, dim=-1, keepdim=True) * scale
    attn = torch.exp(attn)
    attn = attn.clamp(-5, 5)
    if attn_weight is not None:
        # This is for duplicated entries when convert particles to patches
        attn *= attn_weight
    # bs*head, N_r, 1
    normalizer = torch.bmm(r_mask, attn)
    # bs*head, N_e, 1
    e_normalizer = torch.bmm(r_mask.transpose(-1, -2), normalizer)
    attn = attn / (e_normalizer + eps)
    # bs*head, N_e, dim
    e_emb = e_emb * attn
    # bs*head, N_r, dim
    x = torch.bmm(r_mask, e_emb)
    x = x.reshape(bs, num_head, N_r, head_dim)

    return x, attn
