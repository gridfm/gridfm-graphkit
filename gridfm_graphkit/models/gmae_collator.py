import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    # print('-------->', x.size())
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, 
            min_node_num, 
            attn_bias, 
            spatial_pos, 
            in_degree, 
            out_degree, 
            x, 
            y,
            orig_id
            ):
        super(Batch, self).__init__()
        self.min_node_num = int(min_node_num)
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.spatial_pos = attn_bias, spatial_pos
        self.orig_id = orig_id

    def to(self, device):
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.attn_bias, self.spatial_pos = self.attn_bias.to(
            device), self.spatial_pos.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, spatial_pos_max=20):
    """
    custom collator, among other transformations...

    unequal input graphs are padded to all have the same size
    
    adds 1 to the input x via pad_2d_unsqueeze and similar functions
    """
    items = [
        item for item in items if item is not None]
    items = [
            (item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]) 
            for item in items
            ]
    
    # at this step all graphs in batch have their input size
    xs, ys, adjs, attn_biases, spatial_poses, in_degrees, out_degrees, orig_ids = zip(*items)
    
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    min_node_num = min(i.size(0) for i in xs)
    
    if all([torch.all(xx == yy) for xx,yy in zip(xs, ys)]):    # then this is for and encoder-decoder setup
        y = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in ys])
    else:
        y = torch.stack(ys)

    # following steps pad the smaller graphs to match the largest for batching
    # incidentally a constant value of 1 is added as well
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num) for i in attn_biases])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])


    return Batch(
        min_node_num=min_node_num,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        y=y,
        orig_id=orig_ids
    )
