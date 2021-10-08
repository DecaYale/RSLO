
import os
import torch

from torch.utils.cpp_extension import load
import time
import apex.amp as amp
# cd = load(name="cd",
#           sources=["thirdparty/chamfer_distance/chamfer_distance.cpp",
#                    "thirdpartychamfer_distance/chamfer_distance.cu"])

build_directory = os.path.dirname(__file__)+"/build/"

if not os.path.exists(build_directory):
    os.makedirs(build_directory)

cd = load(name="cd",
          sources=[os.path.dirname(__file__)+"/chamfer_distance.cpp",
                   os.path.dirname(__file__)+"/chamfer_distance.cu"],
          build_directory=build_directory)


def chamfer_distance(xyz1, xyz2):
    batchsize, n, _ = xyz1.size()
    _, m, _ = xyz2.size()
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    dist1 = torch.zeros(batchsize, n)
    dist2 = torch.zeros(batchsize, m)

    idx1 = torch.zeros(batchsize, n, dtype=torch.int)
    idx2 = torch.zeros(batchsize, m, dtype=torch.int)

    if not xyz1.is_cuda:
        cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
    else:
        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()
        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

    return dist1, dist2


class ChamferDistanceFunction(torch.autograd.Function):

    @amp.float_function
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @amp.float_function
    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2,
                        graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2,
                             graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistanceFunctionWithIdx(torch.autograd.Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, xyz1, xyz2):
        # dtype = xyz1.dtype
        # xyz1=xyz1.float()
        # xyz2 = xyz2.float()

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2
        # return dist1.to(dtype=dtype), dist2.to(dtype=dtype), idx1, idx2

    @staticmethod
    @amp.float_function
    def backward(ctx, graddist1, graddist2, dummygrad1, dummygrad2):
        # dtype=graddist1.dtype
        # graddist1 = graddist1.float()
        # graddist2 = graddist2.float()
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2,
                        graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2,
                             graddist1, graddist2, idx1, idx2)

        # return gradxyz1.to(dtype=dtype), gradxyz2.to(dtype=dtype)
        return gradxyz1, gradxyz2


class OneDirectionChamferDistanceFunctionWithIdx(torch.autograd.Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, xyz1, xyz2):
        dtype = xyz1.dtype
        xyz1=xyz1.to(dtype=torch.float32)
        xyz2 = xyz2.to(dtype=torch.float32)

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        # dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            raise NotImplementedError
            # cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            # print(xyz1.dtype, dist1.dtype ,flush=True)
            dist1 = dist1.cuda()
            # dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            # idx2 = idx2.cuda()
            cd.forward_cuda_one_direction(
                xyz1, xyz2,
                dist1,
                # dist2,
                idx1,
                # idx2
            )

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, idx1
        # return dist1.to(dtype=dtype), idx1.to(dtype=dtype)

    @staticmethod
    @amp.float_function
    # def backward(ctx, graddist1, graddist2, dummygrad1, dummygrad2):
    def backward(ctx, graddist1,  dummygrad1 ):
        dtype=graddist1.dtype
        graddist1 = graddist1.to(dtype=torch.float32)
        # graddist2 = graddist2.float()
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        # graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), dtype=torch.float32)
        gradxyz2 = torch.zeros(xyz2.size(), dtype=torch.float32)

        if not graddist1.is_cuda:
            # cd.backward(xyz1, xyz2, gradxyz1, gradxyz2,
            #             graddist1, graddist2, idx1, idx2)
            raise NotImplementedError
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
          
            cd.backward_cuda_one_direction(
                xyz1, 
                xyz2, 
                gradxyz1, 
                gradxyz2,
                graddist1, 
                # graddist2, 
                idx1, 
                # idx2
                )

        return gradxyz1.to(dtype=dtype), gradxyz2.to(dtype=dtype)
        # return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)


class ChamferDistanceWithIdx(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunctionWithIdx.apply(xyz1, xyz2)


class OneDirectionChamferDistanceWithIdx(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return OneDirectionChamferDistanceFunctionWithIdx.apply(xyz1, xyz2)
