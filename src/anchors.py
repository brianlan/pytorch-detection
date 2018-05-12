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


def calc_anchor_match(anchors, gt_boxes):
    pass


def get_delta(from_boxes, to_boxes):
    pass


def apply_delta(base_boxes, delta):
    pass
