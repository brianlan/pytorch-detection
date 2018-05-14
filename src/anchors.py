import attr
import numpy as np


@attr.s
class AnchorGenerator(object):
    @staticmethod
    def generate_anchors(fmap_shape, fmap_downsampled_rate, scales, ratios, anchor_stride=1):
        """

        :param fmap_shape:
        :param fmap_downsampled_rate: it's calculated by fmap_shape / im_shape, a concept similar to feature strides.
        :param scales:
        :param ratios:
        :param anchor_stride:
        :return:
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, fmap_shape[0], anchor_stride) / fmap_downsampled_rate
        shifts_x = np.arange(0, fmap_shape[1], anchor_stride) / fmap_downsampled_rate
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes


def calc_overlap(query_boxes, ref_boxes):
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    ref_areas = (ref_boxes[:, 2] - ref_boxes[:, 0]) * (ref_boxes[:, 3] - ref_boxes[:, 1])

    # Compute overlaps to generate matrix [query_boxes count, ref_boxes count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((query_boxes.shape[0], ref_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = ref_boxes[i]
        overlaps[:, i] = calc_iou(box2, query_boxes, ref_areas[i], query_areas)
    return overlaps


def calc_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

        Note: the areas are passed in rather than calculated here for
              efficency. Calculate once in the caller to avoid duplicate work.
        """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def calc_anchor_match(anchors, gt_boxes, fmap_downsampled_rate):
    gt_boxes = gt_boxes * fmap_downsampled_rate
    overlaps = calc_overlap(anchors, gt_boxes)


def get_delta(from_boxes, to_boxes):
    pass


def apply_delta(base_boxes, delta):
    pass
