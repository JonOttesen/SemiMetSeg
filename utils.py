import monai

from t_seg.preprocessing import (
    VolumeInputDropout,
    )

from monai.transforms import (
    CropForegroundd,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandAdjustContrastd,
    RandHistogramShiftd,
    NormalizeIntensityd,
    RandStdShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    SpatialPadd,
    RandZoomd,
)


size = (128, 128, 128)
valid_sizes = (128, 128, 128)
# voxel_sizes = (0.75, 0.75, 0.75)
# voxel_sizes = (1, 1, 1)


transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=3, return_coords=False, mode='constant'),
    
    RandZoomd(keys=["image", "mask"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('trilinear', 'nearest')),
    
    RandRotated(range_x=0.4, range_y=0.4, range_z=0.4, padding_mode='zeros', mode=('bilinear', 'nearest'), keys=['image', 'mask']),
    
    SpatialPadd(keys=["image", "mask"], spatial_size=size, mode='constant', constant_values=0),
    
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=1),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=2),
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3, spatial_axes=(0, 1)),

    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    VolumeInputDropout(keys=["image"], prob=0.25, num_sequences=4, always_keep=[0, 1]),

    RandSpatialCropd(keys=["image", "mask"], roi_size=size, random_center=True, random_size=False),
    ])


semi_transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image"], source_key="image", margin=3, return_coords=False, mode='constant'),
    
    RandZoomd(keys=["image"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('trilinear')),

    SpatialPadd(keys=["image"], spatial_size=size, mode='constant', constant_values=0),

    RandFlipd(keys=["image"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.25, spatial_axis=1),
    RandFlipd(keys=["image"], prob=0.25, spatial_axis=2),
    RandRotate90d(keys=["image"], prob=0.25, max_k=3, spatial_axes=(0, 1)),
    
    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    VolumeInputDropout(keys=["image"], prob=0.25, num_sequences=4, always_keep=[0, 1]),
    RandSpatialCropd(keys=["image"], roi_size=size, random_center=True, random_size=False),
    ])

valid_transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=2, k_divisible=16, return_coords=False, mode='constant'),

    SpatialPadd(keys=["image", "mask"], spatial_size=valid_sizes, mode='constant', constant_values=0),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])



transforms_mix = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=3, return_coords=False, mode='constant'),
    
    RandZoomd(keys=["image", "mask"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('trilinear', 'nearest')),
    
    RandRotated(range_x=0.4, range_y=0.4, range_z=0.4, padding_mode='zeros', mode=('bilinear', 'nearest'), keys=['image', 'mask']),
    
    SpatialPadd(keys=["image", "mask"], spatial_size=size, mode='constant', constant_values=0),
    
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=1),
    RandFlipd(keys=["image", "mask"], prob=0.25, spatial_axis=2),
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3, spatial_axes=(0, 1)),

    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    RandSpatialCropd(keys=["image", "mask"], roi_size=size, random_center=True, random_size=False),
    ])


semi_transforms_mix = monai.transforms.Compose([
    CropForegroundd(keys=["image"], source_key="image", margin=3, return_coords=False, mode='constant'),
    
    RandZoomd(keys=["image"], prob=0.25, min_zoom=0.85, max_zoom=1.25, mode=('trilinear')),

    SpatialPadd(keys=["image"], spatial_size=size, mode='constant', constant_values=0),

    RandFlipd(keys=["image"], prob=0.25, spatial_axis=0),
    RandFlipd(keys=["image"], prob=0.25, spatial_axis=1),
    RandFlipd(keys=["image"], prob=0.25, spatial_axis=2),
    RandRotate90d(keys=["image"], prob=0.25, max_k=3, spatial_axes=(0, 1)),
    
    RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),

    RandSpatialCropd(keys=["image"], roi_size=size, random_center=True, random_size=False),
    ])

valid_transforms = monai.transforms.Compose([
    CropForegroundd(keys=["image", "mask"], source_key="image", margin=2, k_divisible=16, return_coords=False, mode='constant'),

    SpatialPadd(keys=["image", "mask"], spatial_size=valid_sizes, mode='constant', constant_values=0),

    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])