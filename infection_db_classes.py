#!usr/bin/python3

from datetime import datetime

class IdNotFoundException(Exception):
    """ Class that is an exception
    thrown in case no infection or
    outbreak with specified id in class InfectionDB
    """

    def __init__(self, identifier: int, message: str):
        self.id = identifier
        self.message = message.join("ID not found")
        super().__init__(f"{message}: {self.id}")


class Person:
    """ Class being the vertex representing
    a person from the graph in the class InfectionDB
    """
    name: str

    def __init__(self, name: str):
        self.name = name
        self.last_infection = None

    # Method to check if a person is infected
    def is_infected(self) -> bool:
        return bool(self.last_infection)


class Infection:
    """ Class representing the infection
    occurring in class InfectionDB
    """
    infection_id: int
    who: str
    date: datetime
    outbreak_id: int
    direct_infections: int

    def __init__(self, infection_id: int, who_name: str, when: str, source_id: int):
        self.infection_id = infection_id
        self.who = who_name
        self.date = datetime.strptime(when, "%Y-%m-%d")
        self.outbreak_id = source_id
        self.direct_infections = 0

    # Method to increase the number of direct infection-related infections
    def add_direct_infection(self):
        self.direct_infections += 1


class Outbreak:
    """ Class representing an infection
    being the collection of related infections in the class InfectionDB
    """
    outbreak_id: int
    last_date: datetime
    infection_size: int 

    def __init__(self, outbreak_id: int, last_date: str):
        self.outbreak_id = outbreak_id
        self.last_date = datetime.strptime(last_date, "%Y-%m-%d")
        self.infection_size = 1

    # Method to increase the number of infections associated with an outbreak
    def increase_size(self):
        self.infection_size += 1

    # Method to update the date of the latest infection
    def update_date(self, new_date: datetime):
        if new_date > self.last_date:
            self.last_date = new_date

