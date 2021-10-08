import torch 
import torch.nn as nn 
import apex.amp as amp



class SVDHead(nn.Module):
    def __init__(self, args=None):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    @amp.float_function
    def forward(self, src, tgt, weight=None ):
        '''
        src: Bx3xN
        tgt: Bx3xN
        weight: BxN
        '''

        batch_size = src.size(0)

        src_centered = src - src.mean(dim=2, keepdim=True) #Bx3xN
        src_corr_centered = tgt-tgt.mean(dim=2, keepdim=True) #Bx3xN

        if weight is None:
            H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        else:
            H = torch.matmul(src_centered*weight[:,None,:], src_corr_centered.transpose(2, 1).contiguous() )# + torch.eye(3, device=src_centered.device)*1e-9

        # print(H, src_centered.mean(),src_corr_centered.mean(), src_corr_centered.shape)
        U, S, V = [], [], []
        R = []

        #iterate on the batch dimmension
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            # u,s,v=u.cuda(),s.cuda(),v.cuda()
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect.to(device=v.device))
                r = torch.matmul(v, u.transpose(1, 0).contiguous()) # r=v@u^T
                # r = r * self.reflect
            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        # t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)
        #Xs = R@Xt + t
        R = R.transpose(-1,-2).contiguous()
        t = -R@t

        # R = R.to(dtype=dtype)
        # t = t.to(dtype=dtype)
        return R, t.view(batch_size, 3)
