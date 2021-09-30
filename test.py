from collections import OrderedDict


class MyStore:
    """Store keeping track of singleton instances."""

    def __init__(self):
        self.store = {}

    def __str__(self):
        return str({key: str(value) for key, value in self.store.items()})

    def register(self, name, obj):
        self.store[name] = obj


my_store = MyStore()


class MyField(str):
    pass


class MyMeta(type):
    """Example of metaclass demonstrating some of the classical features that such a construct can provide: class
    alteration and registration."""

    @staticmethod
    def __prepare__(class_name: str, class_bases: tuple):
        return OrderedDict()

    def __new__(metacls, class_name: str, class_bases: tuple, class_attrs: OrderedDict):

        breakpoint()

        # Reorganizing attributes:
        reorganized_attrs = OrderedDict([("_fields", OrderedDict()), ("_constants", OrderedDict())])
        for name, attr in class_attrs.items():
            if isinstance(attr, MyField):
                reorganized_attrs["_fields"][name] = attr
            elif not name.startswith("__") and not callable(attr):
                reorganized_attrs["_constants"][name] = attr
            else:
                reorganized_attrs[name] = attr

        # Creating the class:
        cls = type.__new__(metacls, class_name, class_bases, reorganized_attrs)

        # Initializing the singleton pattern:
        obj = cls()
        cls._obj = obj

        # Registering the new object:
        my_store.register(class_name, obj)

        # Displaying the results of the application of the metaclass:
        print(f"Here is what {cls.__name__} contains:")
        for name, attr in cls.__dict__.items():
            print(f"    . {name}: {attr}")
        print("")

        return cls

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_obj"):
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(*args, **kwargs)
            return obj
        else:
            return cls._obj


class MyClass(metaclass=MyMeta):
    """Example of a user-defined (client) class that makes use of MyMeta."""

    # Demo attributes (mixed fields and constants)
    a = 42
    b = MyField("foo")
    c = MyField("bar")
    d = "MyField" in globals()

    def __str__(self):
        """Showing the memory address of self (proving it is a singleton)."""
        return f"I'm located at: {id(self)}"


test_instance = MyClass()
print(test_instance)
other_instance = MyClass()
print("Once again", other_instance)
print("The store is:", my_store)
