from abc import ABC

class InfoGatherField(ABC):
    name: str
    description: str
    
    value: str
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def set_value(self, value: str):
        if self.lint(value):
            self.value = value
    
    def get_value(self) -> str:
        return self.value

    def lint(self, value: str) -> bool:
        return True

class BasicInfoGatherField(InfoGatherField):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)