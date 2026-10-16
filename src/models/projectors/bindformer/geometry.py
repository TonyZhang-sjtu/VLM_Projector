"""Padding-aware support geometry, independent of lesion segmentation.

grid: direction-aware normalized FOV coordinates, volume fraction of the FOV.
voxel: direction-aware voxel coordinates, volume in effective voxel counts.
mm: affine LPS coordinates, volume in mm^3. No atlas registration is implied.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class PatchGeometry:
    centers: torch.Tensor
    volumes: torch.Tensor
    valid_fraction: torch.Tensor
    reference_center: torch.Tensor
    coordinate_scale: float
    volume_scale: float
    space: str


def patch_geometry(coords, metadata, space='grid', coordinate_scale=100., volume_scale=1000.):
    if space not in ('grid', 'voxel', 'mm'):
        raise ValueError('Unknown support geometry space')
    device = coords.device
    order = metadata.get('array_zyx_to_dhw')
    if not isinstance(order, (list, tuple)) or sorted(order) != [0,1,2]:
        raise ValueError('Geometry requires array_zyx_to_dhw')
    bounded = metadata.get('bounded_before_pad', {})
    size = bounded.get('size_xyz')
    before, after = metadata.get('pad_before_xyz'), metadata.get('pad_after_xyz')
    if (size is None or before is None or after is None or
            any(len(v) != 3 for v in (size,before,after)) or
            any(type(n) is not int or n < 1 for n in size) or
            any(type(n) is not int or n < 0 for n in [*before,*after])):
        raise ValueError('Geometry requires valid bounded size and explicit padding metadata')
    direction = bounded.get('direction',metadata.get('oriented',{}).get('direction'))
    if direction is None:
        raise ValueError('Geometry requires image axis direction, even without mm spacing')
    direction = torch.as_tensor(direction,device=device,dtype=torch.float32).reshape(3,3)
    with torch.autocast(device_type=device.type,enabled=False):
        if (not torch.isfinite(direction).all() or
                not torch.allclose(direction.T@direction,torch.eye(3,device=device),rtol=1e-4,atol=1e-4)):
            raise ValueError('Invalid geometry direction')
        size_xyz = torch.tensor(size,device=device,dtype=torch.float32)
        size_dhw = size_xyz.flip(0)[order]
        pad_dhw = torch.tensor(before,device=device,dtype=torch.float32).flip(0)[order]
        patch = torch.tensor([4,16,16],device=device,dtype=torch.float32)
        padded = torch.tensor([s+a+b for s,a,b in zip(size,before,after)],device=device).flip(0)[order]
        grid = metadata.get('grid_dhw')
        if grid is None or not torch.equal(padded,torch.tensor(grid,device=device)*patch):
            raise ValueError('Geometry grid/patch/padding dimensions disagree')
        if (coords.ndim != 2 or coords.shape[1] != 3 or not torch.isfinite(coords).all()
                or (coords < 0).any() or (coords >= torch.tensor(grid,device=device)).any()
                or not torch.equal(coords,coords.round())):
            raise ValueError('Invalid geometry patch coordinates')
        start = coords.float()*patch-pad_dhw
        lo = start.clamp_min(0)
        hi = torch.minimum(start+patch,size_dhw)
        extent = (hi-lo).clamp_min(0)
        voxel_counts = extent.prod(-1)
        if voxel_counts.sum() <= 0:
            raise ValueError('No valid patches for geometry')
        centers_dhw = (lo+hi-1)*.5
        centers_xyz = centers_dhw[:,[order.index(i) for i in range(3)]].flip(-1)
        reference = (size_xyz-1)*.5
        valid_fraction = voxel_counts/math.prod((4,16,16))
        if space == 'grid':
            centers = (2*(centers_xyz+.5)/size_xyz-1)@direction.T
            volumes = voxel_counts/size_xyz.prod()
            reference = torch.zeros(3,device=device)
            coordinate_scale,volume_scale = 1.,1.
        elif space == 'voxel':
            centers,reference = centers_xyz@direction.T,reference@direction.T
            volumes = voxel_counts
        else:
            affine = bounded.get('affine_xyz_to_lps')
            if affine is None:
                raise ValueError('Millimeter geometry requires bounded_before_pad affine')
            affine = torch.as_tensor(affine,device=device,dtype=torch.float32)
            if (affine.shape != (4,4) or not torch.isfinite(affine).all()
                    or not torch.allclose(affine[3],affine.new_tensor([0,0,0,1]))):
                raise ValueError('Invalid physical affine')
            voxel_volume = torch.linalg.det(affine[:3,:3]).abs()
            if voxel_volume <= 0:
                raise ValueError('Singular physical affine')
            # Bounded affine already includes the crop origin; do not add it twice.
            centers = centers_xyz@affine[:3,:3].T+affine[:3,3]
            reference = reference@affine[:3,:3].T+affine[:3,3]
            volumes = voxel_counts*voxel_volume
        centers = torch.where((voxel_counts>0)[:,None],centers,reference[None,:])
    return PatchGeometry(centers,volumes,valid_fraction,reference,
                         coordinate_scale,volume_scale,space)


def support_statistics(assignments, geometry, epsilon=1e-6):
    """[K,10]: standardized center, covariance upper triangle, log soft volume."""
    with torch.autocast(device_type=assignments.device.type,enabled=False):
        a = assignments.float()
        if a.ndim != 2 or a.shape[0] != len(geometry.centers):
            raise ValueError('Geometry support must have shape [N,K]')
        weights = a*geometry.valid_fraction[:,None]
        mass = weights.sum(0)
        normalized = weights/mass.clamp_min(epsilon)[None,:]
        p = (geometry.centers-geometry.reference_center)/geometry.coordinate_scale
        mu = normalized.T@p
        # Compute second moments without allocating an [N,K,3,3] tensor.
        second = normalized.T@(p[:,:,None]*p[:,None,:]).flatten(1)
        covariance = second.reshape(-1,3,3)-mu[:,:,None]*mu[:,None,:]
        diagonal = covariance.diagonal(dim1=-2,dim2=-1)
        covariance = covariance-torch.diag_embed(diagonal)+torch.diag_embed(diagonal.clamp_min(0))
        upper = covariance[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
        volume = (a.T@geometry.volumes)/geometry.volume_scale
        stats = torch.cat((mu,upper,volume.clamp_min(0).log1p()[:,None]),-1)
        usable = mass > epsilon
        return torch.where(usable[:,None],stats,torch.zeros_like(stats)),usable


class SupportGeometryEncoder(nn.Module):
    def __init__(self, dim, hidden_dim=128, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.encoders = nn.ModuleDict({str(r): nn.Sequential(nn.Linear(10,hidden_dim),
                                     nn.GELU(),nn.Linear(hidden_dim,dim)) for r in (1,3)})
        self.dim = dim

    def forward(self, assignments, geometry):
        stats,usable = support_statistics(assignments,geometry,self.epsilon)
        values = []
        for r in range(4):
            if str(r) in self.encoders:
                value = self.encoders[str(r)](stats.to(next(self.parameters()).dtype))
                values.append(value*usable[:,None])
            else:
                values.append(stats.new_zeros((len(stats),self.dim)))
        return torch.stack(values,dim=1)
