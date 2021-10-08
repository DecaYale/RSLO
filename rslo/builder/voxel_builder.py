import numpy as np

# from spconv.utils import VoxelGeneratorV2, VoxelGenerator
from spconv.utils import VoxelGenerator
from rslo.protos import voxel_generator_pb2


# def build(voxel_config):
#     """Builds a tensor dictionary based on the InputReader config.

#     Args:
#         input_reader_config: A input_reader_pb2.InputReader object.

#     Returns:
#         A tensor dict based on the input_reader_config.

#     Raises:
#         ValueError: On invalid input reader proto.
#         ValueError: If no input paths are specified.
#     """
#     if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
#         raise ValueError('input_reader_config not of type '
#                          'input_reader_pb2.InputReader.')
#     voxel_generator = VoxelGeneratorV2(
#         voxel_size=list(voxel_config.voxel_size),
#         point_cloud_range=list(voxel_config.point_cloud_range),
#         max_num_points=voxel_config.max_number_of_points_per_voxel,
#         max_voxels=20000,
#         full_mean=voxel_config.full_empty_part_with_mean,
#         block_filtering=voxel_config.block_filtering,
#         block_factor=voxel_config.block_factor,
#         block_size=voxel_config.block_size,
#         height_threshold=voxel_config.height_threshold)
#     return voxel_generator

class _VoxelGenerator(VoxelGenerator):
    def __init__(self, *args, **kwargs):
        super(_VoxelGenerator, self).__init__(*args, **kwargs)

    @property
    def grid_size(self):
        point_cloud_range = np.array(self.point_cloud_range)
        voxel_size = np.array(self.voxel_size)
        g_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        g_size = np.round(g_size).astype(np.int64)
        return g_size

    def generate(self, points, max_voxels=None):
        res = super(_VoxelGenerator, self).generate(points, max_voxels)

        return {"voxels": res[0],
                "coordinates": res[1],
                "num_points_per_voxel": res[2]
                }


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    # voxel_generator = VoxelGenerator2(
    
    voxel_config.block_filtering=True
    assert(voxel_config.block_filtering)
    if voxel_config.block_filtering:
        voxel_config.block_factor = max(1,voxel_config.block_factor)
        voxel_config.block_size = voxel_config.block_size if voxel_config.block_size>0 else 8

        voxel_config.height_threshold = voxel_config.height_threshold if voxel_config.height_threshold!=0 else 0.2

    voxel_generator = _VoxelGenerator(
        voxel_size=list(voxel_config.voxel_size),
        point_cloud_range=list(voxel_config.point_cloud_range),
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        max_voxels=20000,
        full_mean=False,
        # full_mean=voxel_config.full_empty_part_with_mean,
        block_filtering=voxel_config.block_filtering,
        block_factor=voxel_config.block_factor,
        block_size=voxel_config.block_size,
        height_threshold=voxel_config.height_threshold
    )
    return voxel_generator
