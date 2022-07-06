from src.detection.utils.ssd_utils import get_default_box_count
from src.detection.utils.bbox_utils import center_to_corner, corner_to_center
import numpy as np


def generate_default_boxes_for_feature_map(
        feature_map_size,
        image_size,
        offset,
        scale,
        next_scale,
        aspect_ratios,
        variances,
        has_extra_box_for_ar_1,
        clip_boxes=True
):
    '''
    Generates a 4D tensor representing default boxes [xmin, ymin, xmax, ymax] for a given feature map
    :param feature_map_size: Size of square feature map
    :param image_size: Size of square input image
    :param offset: Offset for center of default boxes in order: (offset_x, offset_y)
    :param scale: Current scale of default boxes
    :param next_scale: Next scale of default boxes
    :param aspect_ratios: List of aspect ratios representing default boxes
    :param variances: Variances (x, y, w, h) applied to generated priors
    :param has_extra_box_for_ar_1: Boolean for determining presence of extra box for aspect ratio 1
    :param clip_boxes: Boolean for determining whether to clip output default boxes
    '''

    assert (len(offset) == 2, 'Parameter: "offset" must have len of 2')

    grid_size = image_size / feature_map_size
    offset_x, offset_y = offset
    default_box_count = get_default_box_count(aspect_ratios, has_extra_box_for_ar_1)

    # Calculate width and height of default boxes
    default_box_wh_list = []
    for ar in aspect_ratios:
        # SSD authors add additional default box with special scale if aspect ratio is 1.0
        if ar == 1.0 and has_extra_box_for_ar_1:
            default_box_wh_list.append([
                image_size * np.sqrt(scale * next_scale) * np.sqrt(ar),
                image_size * np.sqrt(scale * next_scale) * (1 / np.sqrt(ar))
            ])
        default_box_wh_list.append([
            image_size * scale * np.sqrt(ar),  # Width formula based on original SSD paper
            image_size * scale * (1 / np.sqrt(ar))  # Height formula based on original SSD paper
        ])
    default_box_wh_list = np.array(default_box_wh_list, dtype=np.float)

    # Calculate center points of each grid cell
    cx = np.linspace(offset_x * grid_size, image_size - (offset_x * grid_size), feature_map_size)
    cy = np.linspace(offset_y * grid_size, image_size - (offset_y * grid_size), feature_map_size)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid, cy_grid = np.expand_dims(cx_grid, axis=-1), np.expand_dims(cy_grid, axis=-1)
    cx_grid, cy_grid = np.tile(cx_grid, (1, 1, default_box_count)), np.tile(cy_grid, (1, 1, default_box_count))

    default_boxes = np.zeros((feature_map_size, feature_map_size, default_box_count, 4))
    default_boxes[..., 0] = cx_grid
    default_boxes[..., 1] = cy_grid
    default_boxes[..., 2] = default_box_wh_list[:, 0]
    default_boxes[..., 3] = default_box_wh_list[:, 1]

    # Clip overflow default boxes
    if clip_boxes:
        default_boxes = center_to_corner(default_boxes)
        for coord_indices in ([0, 2], [1, 3]):  # Clipping on ([xmin, xmax], [ymin, ymax])
            coords = default_boxes[..., coord_indices]
            coords[coords >= image_size] = image_size - 1
            coords[coords < 0] = 0
            default_boxes[..., coord_indices] = coords
        default_boxes = corner_to_center(default_boxes)

    default_boxes /= image_size

    variances_tensor = np.zeros_like(default_boxes)
    variances_tensor += variances
    default_boxes = np.concatenate([default_boxes, variances_tensor], axis=-1)

    return default_boxes
