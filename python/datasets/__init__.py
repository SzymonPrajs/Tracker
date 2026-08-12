"""One readable adapter per downloaded dataset."""

from . import crowdhuman, open_images, scut_head, wider_face

ALL = {
    "wider_face": wider_face.run,
    "scut_head": scut_head.run,
    "crowdhuman": crowdhuman.run,
    "open_images": open_images.run,
}
