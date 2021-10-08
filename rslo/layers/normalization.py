import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import apex
from apex.parallel.sync_batchnorm_kernel import SyncBatchnormFunction
from apex.parallel import ReduceOp


class MaskSyncBatchNorm(apex.parallel.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, process_group=process_group, channel_last=channel_last)

        # self.const_scale =0.5
        # print("MaskSyncBN const_scale=", self.const_scale)

    
    def forward(self, input):
        input,mask = input
        # mask = (input.sum(dim=list(range(2,len(input.shape))), keepdim=True)!=0).float().detach() #Bx1xHxW
        mask = (mask[:,0:1]>0).float().detach() # Bx1xHxW
        input*=mask
        valid_num = mask.sum(dim=list(range(2,len(input.shape))) ).sum() # Bx1->1
        input_num = input.shape[0] 
        for s in input.shape[2:]:
            input_num *= s
        rectifier = input_num/(valid_num+1e-3) #* self.const_scale #!!!
        # print(valid_num, input_num, rectifier,'!!')
        # mask_bool = mask.to(dtype=torch.uint8).expand_as(input).detach()

        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        cast = None
        out = None

        # casting to handle mismatch input type to layer type
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype

        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
            # out *=mask
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():

                channel_first_input = input.transpose(0, 1).contiguous()
                # channel_first_mask= mask_bool.transpose(0,1).contiguous()

                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                # squashed_input_tensor_view = channel_first_input[channel_first_mask].view(channel_first_input.size(0), -1)

                # total number of data points for each variance entry. Used to calculate unbiased variance estimate
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                # local_mean = torch.mean(squashed_input_tensor_view, 1)
                # local_sqr_mean = torch.pow(
                #     squashed_input_tensor_view, 2).mean(1)
                
                local_mean = torch.mean(squashed_input_tensor_view, 1)*rectifier
                local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)*rectifier**2
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(
                        local_mean, ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(
                        local_sqr_mean, ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)
                

                if self.running_mean is not None:
                    self.running_mean = self.momentum * mean + \
                        (1 - self.momentum) * self.running_mean
                if self.running_var is not None:
                    # as noted by the paper, we used unbiased variance estimate of the mini-batch
                    # Var[x] = m / (m-1) * Eb (sample_variance)
                    self.running_var = m / \
                        (m-1) * self.momentum * var + \
                        (1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
            # out*=mask
        return out.to(cast)


class SemiGlobalSyncBatchNorm(apex.parallel.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, process_group=process_group, channel_last=channel_last)

        # self.first_pass=True
        self.momentum=0.9
        self.avg_runing=False#True
        self.dyn_mom=True
        print("mom, avg, dyn_mom:",self.momentum, self.avg_runing, self.dyn_mom)
        self.iter_cnt=0
        self.register_buffer('mean_dyn_mom', torch.full((num_features,),momentum, ))
        self.register_buffer('var_dyn_mom', torch.full((num_features,),momentum, ))
        self.register_buffer('running_mean_g2', torch.ones((num_features,), dtype=torch.float32) )
        self.register_buffer('running_var_g2', torch.ones((num_features,), dtype=torch.float32) )
        self.register_buffer('running_mean_probe', torch.zeros((num_features,), dtype=torch.float32) )
        self.register_buffer('running_var_probe', torch.ones((num_features,), dtype=torch.float32) )


    # def update_momentum_and_g2(self,dyn_mom, running_g2, running_old, running_new, beta=0.1):
    def update_momentum_and_g2(self,dyn_mom, running_g2, running_probe, val, beta=0.1):
        running_probe_old = running_probe
        running_probe = (1-beta)*running_probe + beta*val # used for probing the stability of val 

        # diff = (running_new-running_old)/running_old
        diff = (running_probe-running_probe_old)/running_probe_old
    
        diff = diff**2
        # print(diff.sum(),'!!')
        
        running_g2 = (1-beta)*running_g2 + beta*diff 
        # running_g2 = torch.clamp(running_g2, max=1, min=1-self.momentum) # [0,1]
        running_g2 = torch.clamp(running_g2, max=self.momentum**2, min=0) # [0,1]
        # running_g2 = torch.max(torch.min(running_g2, (1-dyn_mom)/(1-self.momentum)), 1-dyn_mom) # [0,1]
        dyn_mom = 1-(1-self.momentum)/(1-self.momentum+torch.sqrt(running_g2)+1e-9) 

        return dyn_mom, running_g2, running_probe


    def forward(self, input):
        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        cast = None
        out = None

        # casting to handle mismatch input type to layer type
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype

        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                # total number of data points for each variance entry. Used to calculate unbiased variance estimate
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(
                        local_mean, ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(
                        local_sqr_mean, ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)

                # if self.first_pass:
                #     print("First pass of SemiGlobalSyncBatchNorm.")
                #     self.first_pass=False
                #     self.running_mean=mean
                #     self.running_var = var
                if self.iter_cnt<1/self.momentum:
                    self.iter_cnt+=1
                if self.running_mean is not None:
                    
                    if not self.avg_runing:
                        if not self.dyn_mom:
                            self.running_mean = self.momentum * mean + \
                                (1 - self.momentum) * self.running_mean.detach()
                        else:
                            momentum=self.mean_dyn_mom
                            self.running_mean_old = self.running_mean
                            self.running_mean = momentum * mean + \
                                (1 - momentum) * self.running_mean.detach()
                            self.mean_dyn_mom, self.running_mean_g2, self.running_mean_probe=self.update_momentum_and_g2(self.mean_dyn_mom, self.running_mean_g2,running_probe=self.running_mean_probe, val=mean)
                            # print(self.mean_dyn_mom[0], self.running_mean_g2[0],'!')

                    else:
                        self.running_mean = mean/self.iter_cnt+self.running_mean.detach()*(1-1/self.iter_cnt)

                if self.running_var is not None:
                    # as noted by the paper, we used unbiased variance estimate of the mini-batch
                    # Var[x] = m / (m-1) * Eb (sample_variance)
                    if not self.avg_runing:
                        if not self.dyn_mom:
                            self.running_var = m / \
                                (m-1) * self.momentum * var + \
                                (1 - self.momentum) * self.running_var.detach()
                        else:
                            momentum=self.var_dyn_mom
                            self.running_var_old = self.running_var
                            self.running_var = m / \
                                (m-1) * momentum * var + \
                                (1 - momentum) * self.running_var.detach()
                            self.var_dyn_mom, self.running_var_g2, self.running_var_probe=self.update_momentum_and_g2(self.var_dyn_mom, self.running_var_g2,running_probe=self.running_var_probe, val=var)
                            # print(self.var_dyn_mom[0], self.running_var_g2[0],'!＝')
                    else:
                        self.running_var = var/self.iter_cnt+self.running_var.detach()*(1-1/self.iter_cnt)
            torch.cuda.nvtx.range_pop()
            # out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, self.running_mean.detach(), self.running_var.detach(), self.eps, process_group, world_size)
        return out.to(cast)




class SparseInstanceNorm1d(nn.InstanceNorm1d):
    # @weak_script_method
    def forward(self, input):
        raise NotImplementedError()
        # input = torch.unsqueeze(input, 2)
        # input = super(SparseInstanceNorm1d,self).forward(input)
        # return torch.squeeze(input,2)


class SpatialGroupedInstanceNorm2d(nn.Module):
    #NxCxHxW
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SpatialGroupedInstanceNorm2d,self).__init__()

        assert len(num_groups) == 2 and (num_groups[0]==1 or num_groups[1]==1) 
        self.num_groups = num_groups
        self.groups = num_groups[0]+num_groups[1]-1
        self.eps= eps
        self.affine=affine

        if self.affine:
            self.weight = Parameter(torch.zeros([self.groups, num_channels]))
            self.bias = Parameter(torch.zeros([self.groups, num_channels] ))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    # @weak_script_method
    def forward(self, input):
        if self.num_groups[0] > 1:
            x = input.permute([0,1,3,2]) # NxCxHxW->NxCxWxH
        else:
            x = input

        N,C,H,W = x.shape
        
        # NxCxHxW -> NxHxWxC
        x = x.permute([0,2,3,1])
        group_size= W//self.groups#(W+self.groups-1)//self.groups
        std_group_num = self.groups if W%self.groups==0 else self.groups-1

        last_group_size=W-std_group_num*group_size#W-group_size* std_group_num
        
        # print(group_size, std_group_num,last_group_size,'!!!')
        x_first = x[:,:,:W-last_group_size].reshape([N,H,std_group_num, -1, C])

        mean_first = torch.mean(x_first, dim=3, keepdim=True) 
        var_first = torch.var(x_first, dim=3, keepdim=True, unbiased=False)
        # import pdb 
        # pdb.set_trace() 
        x_first = (x_first-mean_first)/torch.sqrt(var_first+self.eps) * self.weight[:std_group_num].unsqueeze(1) +self.bias[:std_group_num].unsqueeze(1) 
        x_first  = x_first.reshape([N,H,W-last_group_size,C])
        if last_group_size > 0:
            x_last =x[:,:,W-last_group_size:].reshape([N,H,1, -1, C]) 
            mean_last = torch.mean(x_last, dim=3, keepdim=True) 
            var_last = torch.var(x_last, dim=3, keepdim=True, unbiased=False)
            x_last = (x_last-mean_last)/torch.sqrt(var_last+self.eps) * self.weight[std_group_num:].unsqueeze(1) +self.bias[std_group_num:].unsqueeze(1) 
            x_last = x_last.reshape([N,H,last_group_size, C]) 
        else:
            x_last = torch.Tensor([], device=x_first.device) 

        x = torch.cat([x_first, x_last], dim=2).permute(0,3,1,2)

        if self.num_groups[0] > 1:
            x = x.permute([0,1,3,2]) # NxCxWxH-> NxCxHxW
       
        return x 

        # return F.group_norm(
        #     input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
