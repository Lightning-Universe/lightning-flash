from typing import Any, Mapping, Sequence, Union

from torch import nn

from flash.data.utils import convert_to_modules


class ApplyToKeys(nn.Sequential):

    def __init__(self, keys: Union[str, Sequence[str]], *args):
        super().__init__(*[convert_to_modules(arg) for arg in args])
        if str(keys) == keys:
            keys = [keys]
        self.keys = keys

    def forward(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
        inputs = [x[key] for key in filter(lambda key: key in x, self.keys)]
        if len(inputs) > 0:
            outputs = super().forward(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs, )

            result = {}
            result.update(x)
            for i, key in enumerate(self.keys):
                result[key] = outputs[i]
            return result
        return x
