from flash.core.adapter import Adapter


class PytorchTabularAdapter(Adapter):
    @classmethod
    def from_task(cls, task: "flash.Task", **kwargs) -> "Adapter":
        pass
