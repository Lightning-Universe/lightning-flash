import abc
from typing import get_type_hints

from flash.core.serve._compat import cached_property


class BaseType(metaclass=abc.ABCMeta):
    """Base class for Types.

    Any Grid Types must be inherited from this class and must implement abstract
    methods. The constructor (or the initializer for pythonistas) can take parameters
    and customize the behaviour of serialization/deserialization process.

    Notes
    -----
    *  The argument to the :meth:`deserialize` method must be type annotated. This
       information will be used to construct the API endpoint. For instance, if you are
       making a custom ``Text`` type, you might expect the end user to pass text string
       and the language, you could define this method like this:

       .. code-block:: python

          def deserialize(self, text: str, language: str):
              pass

    *  This will be translated to an API endpoint (automatically and transparently -
       no explicit coding required from you) that takes a dictionary that would look
       like this:

       .. code-block:: python

          {"text": "some string", "language": "en"}
    """

    @cached_property
    def type_hints(self):
        """Fetch the output hints from serialize and input hints from deserialize."""

        input_types = get_type_hints(self.deserialize)
        input_types.pop("return", None)
        try:
            output_types = get_type_hints(self.serialize)["return"]
        except KeyError:  # pragma: no cover
            raise RuntimeError("Did you forget to type annotate " "the `serialize` method?")
        return {"output_args": output_types, "input_args": input_types}

    @abc.abstractmethod
    def serialize(self, data):  # pragma: no cover
        """Serialize the incoming data to send it through the network."""
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize(self, *args, **kwargs):  # pragma: no cover
        """Take the inputs from the network and deserialize/convert them.

        Output from this method will go to the exposed method as arguments.
        """
        raise NotImplementedError

    def packed_deserialize(self, kwargs):
        """Unpacks data (assuming kwargs) and calls deserialize method of child class.

        While it does not seem to be doing much, and always does one thing, the benefit comes when building
        sophisticated datatypes (such as Repeated) where the developer wants to dictate how the unpacking happens. For
        simple cases like Image or Bbox etc., developer would never need to know the existence of this. Task graph would
        never call deserialize directly but always call this method.
        """
        return self.deserialize(**kwargs)
