class Registry:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name):
        """Decorator to register a class/function with a given name."""

        def decorator(cls):
            if name in self._registry:
                raise KeyError(f"{name} already registered in {self._name}")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name]

    def create(self, name, *args, **kwargs):
        """Instantiate a registered class/function with given args."""
        cls = self.get(name)
        return cls(*args, **kwargs)

    def __str__(self):
        return f"Registry<{self._name}>({list(self._registry.keys())})"

    def __contains__(self, key):
        return key in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)


MASKING_REGISTRY = Registry("mask")
NORMALIZERS_REGISTRY = Registry("norm")
MODELS_REGISTRY = Registry("model")
