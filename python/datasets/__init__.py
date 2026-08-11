"""One readable adapter per downloaded dataset."""

from . import coco, crowdhuman, scut_head, wider_face

ALL = {
    "wider_face": wider_face.run,
    "scut_head": scut_head.run,
    "crowdhuman": crowdhuman.run,
    "coco": coco.run,
}
