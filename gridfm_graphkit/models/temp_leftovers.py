# temporary file to hold functions while they wait to be 
# transferred to other modules


    self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
    self.in_degree_encoder = nn.Embedding(
        512, hidden_dim, padding_idx=0)
    self.out_degree_encoder = nn.Embedding(
        512, hidden_dim, padding_idx=0)

    
    def compute_pos_embeddings(self, batched_data):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        # graph_attn_bias
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node, n_node]
        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        node_feature = self.input_proj(x)
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_node_feature = node_feature

        return graph_node_feature, graph_attn_bias


    def training_step(self, batched_data, batch_idx):
        num_nodes = batched_data.x.size(1)

        # create a boolean mask where padding was added
        # note that this assumes all input data had features with
        # values >= 0
        mask = None
        masked_entries = torch.sum(batched_data.x == 0, axis=2)
        mask = masked_entries == batched_data.x.size(2)

        # add low-level random noise to input X
        noise = np.random.normal(
                    loc=0.0,
                    scale=0.00001,  # TODO make configurable
                    size=batched_data.x.size()
        )
        device = batched_data.x.device
        orig_data = batched_data.x
        batched_data.x = batched_data.x + torch.Tensor(noise).to(device)

        strategy = ''
        # fifty-fifty split between random masking and power-flow solution
        if np.random.uniform() > 0.5:
            # find location of all nozero entries for masking and shuffle, select, mask
            inds = torch.where(orig_data.flatten() != 0)
            num_mask = int(self.mask_ratio * len(inds[0]))
            shuf_inds = (inds[0][torch.randperm(len(inds[0]))],)

            nshape = batched_data.x.size()
            batched_data.x = batched_data.x.flatten()
            batched_data.x[shuf_inds[0][:num_mask].to(device)] = self.masking_value
            batched_data.x = torch.reshape(batched_data.x, nshape)
        else:   # assume only  voltage and power variables to be masked
            inds = torch.cat([
                    # to pred
                    torch.range(xx,len(orig_data.flatten()), 25, dtype=int) 
                    for xx in [ii for ii in range(17,25)]
                ])

            shuf_inds = inds[torch.randperm(len(inds))]

            nshape = batched_data.x.size()
            batched_data.x = batched_data.x.flatten()
            batched_data.x[shuf_inds.to(device)] = self.masking_value
            batched_data.x = torch.reshape(batched_data.x, nshape)

        
        y_hat, graph_mask = self(batched_data, mask)  # [n_graph, n_masked_node, n_feature]
        if graph_mask is not None:
            y_gt = orig_data[graph_mask].float()
        else:
            y_gt = orig_data.float()

        y_gt = y_gt[~mask]
        y_hat = y_hat[~mask]

        # print('pre loss shapes', y_gt.size(), y_hat.size())
        loss = self.loss_fn(y_hat, y_gt)
        loss_actv = self.alpha*self.loss_phys1(y_hat, y_gt, device)
        self.log('train_loss', loss)
        self.log('activ_loss', loss_actv)

        return loss + loss_actv

    def validation_step(self, batched_data, batch_idx):
        num_nodes = batched_data.x.size(1)
        mask = None

        masked_entries = torch.sum(batched_data.x == 0, axis=2)
        mask = masked_entries == batched_data.x.size(2)

        # add low-level random noise to input X
        noise = np.random.normal(
                    loc=0.0,
                    scale=0.00001,  # TODO make configurable
                    size=batched_data.x.size()
        )
        device = batched_data.x.device
        orig_data = batched_data.x
        batched_data.x = batched_data.x + torch.Tensor(noise).to(device)
        
        # fifty-fifty split between random masking and power-flow solution
        if np.random.uniform() > 0.5:
            # find location of all nozero entries for masking and shuffle, select, mask
            inds = torch.where(orig_data.flatten() != 0)
            num_mask = int(self.mask_ratio * len(inds[0]))
            shuf_inds = (inds[0][torch.randperm(len(inds[0]))],)

            nshape = batched_data.x.size()
            batched_data.x = batched_data.x.flatten()
            batched_data.x[shuf_inds[0][:num_mask].to(device)] = self.masking_value
            batched_data.x = torch.reshape(batched_data.x, nshape)
        else:   # assume only  voltage and power variables to be masked
            inds = torch.cat([
                    # to pred
                    torch.range(xx,len(orig_data.flatten()), 25, dtype=int) 
                    for xx in [ii for ii in range(17,25)]
                ])

            shuf_inds = inds[torch.randperm(len(inds))]

            nshape = batched_data.x.size()
            batched_data.x = batched_data.x.flatten()
            batched_data.x[shuf_inds.to(device)] = self.masking_value
            batched_data.x = torch.reshape(batched_data.x, nshape)
        
        y_hat, graph_mask = self(batched_data, mask)  # [n_graph, n_masked_node, n_feature]
        if graph_mask is not None:
            y_gt = orig_data[graph_mask].float()
        else:
            y_gt = orig_data.float()

        no_features = y_hat.size(2)
        y_gt = y_gt[~mask]
        y_hat = y_hat[~mask]
        y_hat = y_hat.reshape(-1, y_hat.size(1))  # [n_graph*n_masked_node, n_feature]
        y_gt = y_gt.reshape(-1, y_gt.size(1))  # [n_graph*n_masked_node, n_feature]
        pad_mask = torch.nonzero(y_gt.sum(-1))
        
        y_gt = y_gt[pad_mask, :]
        y_hat = y_hat[pad_mask, :]

        loss = self.loss_fn(y_hat, y_gt)
        loss_actv = self.alpha*self.loss_phys1(y_hat, y_gt, device)
        self.log('val_loss', loss, batch_size=1)

        # loss per feature, for logging only
        for ii in range(no_features):
            self.log(
                    'val_loss_{}'.format(ii), 
                    self.loss_fn(y_hat[ii::no_features], y_gt[ii::no_features]), 
                    batch_size=1
                    )

        return loss + loss_actv