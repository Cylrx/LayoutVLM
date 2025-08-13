import os
import sys

_this_dir = os.path.dirname(__file__)
_rotated_iou_dir = os.path.join(_this_dir, "Rotated_IoU")
if os.path.isdir(_rotated_iou_dir) and _rotated_iou_dir not in sys.path:
    sys.path.append(_rotated_iou_dir)


