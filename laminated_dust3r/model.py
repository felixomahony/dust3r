from typing import Any
from dust3r.model import AsymmetricCroCo3DStereo


class LaminatedDust3rModel:
    def __init__(self, model):
        self._model = model

    @classmethod
    def default(cls, device="cuda"):
        weights_path = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        return cls(
            model=AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._model(*args, **kwds)


if __name__ == "__main__":
    m: LaminatedDust3rModel = LaminatedDust3rModel.default(device="cpu")
