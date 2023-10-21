

class Sensor:
    def __init__(self, name: str, parents=set(), initial_data=None) -> None:
        self.name = name
        self.data = initial_data
        self.parents = parents

    def sense(self):
        pass

class Actuator:
    def __init__(self, name: str, parents=set(), receptors=set()) -> None:
        self.name = name
        self.parents = parents

    def act(self):
        pass

class Controller:
    def __init__(self, name: str, sensors=set(), actuators=set(), *aux) -> None:
        self.name = name
        self.sensors = sensors
        self.actuators = actuators
        self.aux = aux
    