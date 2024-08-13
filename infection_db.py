#!usr/bin/python3

"""
The file contains support classes
"""
import heapq
from math import log
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from infection_db_classes import IdNotFoundException, Person,\
                                 Infection, Outbreak

class InfectionDB:
    """Class containing information on infections
    transmitted between persons according to a graph of mutual relationships.
    The graph is undirected, represented by persons as nodes
    and intensities as weighted edges. Furthermore, the weights of all edges
    are natural numbers less than a predetermined parameter dmax
    """
    people_names: List[str]
    start_day: datetime
    people: List[Person]
    dmax: int
    outbreaks: Dict[int, Outbreak]
    infections: List[Infection]

    def __init__(self, start_day: str, people_names: List[str],\
                connections: Dict[Tuple[str, str], int], dmax: int):
        """ Constructor creates a data structure
        for the infection database based on a graph G(V=people_names, E=connections)
        with the start date of data collection, parameter dmax, counts of
        infections and outbreaks occurring.
        Time complexity: O(|V| + |E|)
        Memory complexity: O(|V| + |E|)
        """
        self.people_names = people_names
        self.start_day = datetime.strptime(start_day, "%Y-%m-%d")
        self.people = {name: Person(name) for name in self.people_names} # Dictionary of nodes
        self.dmax = dmax
        self.outbreaks = {}
        self.outbreaks_count = 0
        self.infections = []
        self.infections_count = 0
        self.largest_outbreak = None

        self.connections = defaultdict(list) # Dictionary of adjacent vertices
        for (u, v), w in connections.items():
            self.connections[u].append((v, w))
            self.connections[v].append((u, w))

        self.direct_infections_connections = defaultdict(set) # Dictionary of direct infections

        # Identifying an efficient algorithm for determining shortest paths
        # Comparison through time complexity of the Djikstra and Dial algorithm.
        if (len(connections) / dmax) > (len(people_names) / log(len(people_names), 2)):
            self.djikstra = False
        else:
            self.djikstra = True

    def number_of_outbreaks(self) -> int:
        """ Mehod returns the number of all outbreaks in the infection datebase.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        return self.outbreaks_count

    def number_of_direct_infections(self, infection_id: int) -> int:
        """ Method returns the number of persons infected directly from the
        given case identified by infection_id, if
        is not present it throws an IdNotFoundException exception.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        if infection_id >= self.infections_count:
            raise IdNotFoundException(infection_id, "Infection")
        return self.infections[infection_id].direct_infections

    def number_of_outbreak_infections(self, outbreak_id: int) -> int:
        """ Method returns the number of infected in the outbreak identified by
        outbreak_id (directly and indirectly) of infections in the specified
        outbreak, if no found it throws an IdNotFoundException.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        try:
            return self.outbreaks[outbreak_id].infection_size
        except KeyError:
            raise IdNotFoundException(outbreak_id, "Outbreak")

    def outbreak_id_of_infection(self, infection_id: int) -> int:
        """ Method returns the id of the outbreak identified by infection_id,
        throws an IdNotFoundException if not found.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        if infection_id >= self.infections_count:
            raise IdNotFoundException(infection_id, "Infection")
        return self.infections[infection_id].outbreak_id

    def largest_outbreak_id(self) -> int | None:
        """ Method returns the identifier of the outbreak with the highest
        number of infections(indirect and direct) in the entire database or None.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        return self.largest_outbreak

    def outbreak_is_active(self, outbreak_id: int, when: str) -> bool:
        """The method checks whether new infections might occur in the
        indicated outbreak (outbreak_id) after the indicated time (when)
        assuming that no new outbreaks have occurred in this outbreak by that time.
        The parameter when is of the form "%Y-%m-%d".
        If outbreak_id is missing, it throws an IdNotFoundException.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        if outbreak_id >= self.outbreaks_count:
            raise IdNotFoundException(infection_id, "Outbreak")
        return self.outbreaks[outbreak_id].last_date + timedelta(days=self.dmax) >= \
                   datetime.strptime(when, "%Y-%m-%d")

    def __increase_outbreak(self, when: str, outbreak_id: int) -> None:
        """ Private method that updates a given outbreak.
        In an outbreak with the given outbreak_id, the date of the last
        infection is changed and the its size is increased by one.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        self.outbreaks[outbreak_id].update_date(datetime.strptime(when, "%Y-%m-%d"))
        self.outbreaks[outbreak_id].increase_size()
        if self.outbreaks[outbreak_id].infection_size > \
            self.outbreaks[self.largest_outbreak].infection_size:
            self.largest_outbreak = outbreak_id

    def __add_new_infection(self, who_name: str, when: str, outbreak_id: int ) -> Infection:
        """ Private method adding a new infection
        to the stored database. Returns the created infection.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        new_infection = Infection(self.infections_count, who_name, when, outbreak_id)
        self.infections_count += 1
        self.infections.append(new_infection)
        self.people[who_name].last_infection = new_infection
        return new_infection

    def add_infection(self, who_name: str, when: str, source_id=None) -> int:
        """ Method creates a new infection case and adds it to the database.
        The new infection has the specified infection source, which can be another infection
        with the specified source_id, or if no source is given 
        creates a new outbreak, the first case of which will be the newly created infection.
        The 'when' parameter is of the form "%Y-%m-%d", e.g. "2024-05-05".
        The function returns the identifier of the added infection case.
        Time complexity: O(1)
        Memory complexity: O(1)
        """
        # Increasing the number of infections
        infection_id = self.infections_count

        if source_id is None or source_id >= self.infections_count:
            source_id = self.infections_count
            self.outbreaks_count += 1
            new_outbreak = Outbreak(infection_id, when) # Stworzenie nowego ogniska
            self.outbreaks[infection_id] = new_outbreak
            if self.largest_outbreak is None:
                self.largest_outbreak = infection_id
        else:
            source_infection = self.infections[source_id]
            source_infection.add_direct_infection() # Dodanie bezpośredniego zakażenia
            source_id = source_infection.outbreak_id
            self.__increase_outbreak(when, source_id)
            self.direct_infections_connections[source_infection.who].add(who_name)

        self.__add_new_infection(who_name, when, source_id)
        return infection_id

    def djikstra_shortest_path(self, who_name: str) -> Dict[str, int]:
        """ Dijkstra algorithm based on storing visited vertices
        in a binary heap. This ensures that the removal and addition of vertices
        (heappop and heappush) is done in O(logV) complexity. It is also necessary
        to traverse each edge and visit neighbouring vertices with a greedy
        alorithm with an upper bound of dmax.
        Time complexity: O(E*logV)
        Memory complexity: O(V + E)
        """
        distances = {} # distance dictionary
        heap = [(0, who_name)] # heap containing visited vertices

        while heap:
            distance, node = heapq.heappop(heap)
            if node in distances:
                continue

            distances[node] = distance
            for neighbor, weight in self.connections[node]:
                new_distance = distance + weight
                if neighbor not in distances and new_distance <= self.dmax:
                    heapq.heappush(heap, (new_distance, neighbor))

        return distances

    def dial_shortest_path(self, who_name: str) -> Dict[str, int]:
        """ Dial algorithm which is an optimisation of the Dijkstra algorithm.
        It is efficient for graphs in which the edge weights are small,
        bounded integers. In this case, the constraint is dmax, one of the
        attributes of our class. We use bins to store vertices according to
        their distances. The maximum number of bins that we search is dmax * V
        and thanks to the priority queue we can remove the first elements
        in O(1) time complexity. The time optimisation is due to the fact that
        the length of the shortest path between two vertices in a graph with
        constrained weights is dmax * (|V| - 1).
        Code adapted from https://www.geeksforgeeks.org/dials-algorithm-optimized-dijkstra-for-small-range-weights/
        Time complexity: O(dmax*V + E)
        Memory complexity: O(V + E)
        """
        max_distance = self.dmax + 1
        distances = {person: float("inf") for person in self.people_names}
        distances[who_name] = 0
        buckets = [deque() for _ in range(max_distance)]
        buckets[0].append(who_name)

        # Searching the graph until dmax is reached
        for bucket_index in range(max_distance):

            while buckets[bucket_index]:
                current_person = buckets[bucket_index].popleft()
                current_distance = distances[current_person]

                # Processing neighbours
                for neighbor, weight in self.connections[current_person]:
                    new_distance = current_distance + weight
                    if new_distance < max_distance:
                        buckets[new_distance].append(neighbor)
                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance

        return distances

    def potential_infection_sources(self, who_name: str, when: str) -> List[str]:
        """ Method returns the set of people from whom the indicated person
        can become infected at a given time (when). The when parameter has
        the form "%Y-%m-%d". It is necessary to search the graph for the shortest paths
        leading away from the selected person, and then check which persons are infected.
        The choice of algorithm depends on the number of edges, vertices and dmax.
        Time complexity: O(min(dmax*V + E, E*logV + V))
        Memory complexity: O(V + E)
        """
        if self.djikstra:
            distances = self.djikstra_shortest_path(who_name)
        else:
            distances = self.dial_shortest_path(who_name)
        return [person for person, distance in distances.items() if 
                self.people[person].is_infected() and
                self.people[person].last_infection.date + timedelta(days=distances[person]) <
                datetime.strptime(when, "%Y-%m-%d") <=
                self.people[person].last_infection.date + timedelta(days=self.dmax)]

    def add_infection_inferred_source(self, who_name: str, when: str) -> int:
        """Metoda tworzy i dodaje do bazy nowy przypadek zakażenia o nieznanym źródle. 
        Parametr when ma postać "%Y-%m-%d". Zwraca identyfikator dodanego przypadku zakażenia.
        W tym celu należy przejść cały graf (metodą potential_infection_sources) oraz znaleźć
        najwcześniejsze źródło zakeżenia, dla którego dany wierzchołek jest zainfekowany.
        Złożoność czasowa: O(min(dmax*V + E, E*logV + V))
        Złożoność pamięciowa: O(V + E)
        """
        inferred_sources = self.potential_infection_sources(who_name, when)
        if len(inferred_sources) == 0:
            return self.add_infection(who_name, when, None)
        else:
            inferred_infection = self.people[inferred_sources[0]].last_infection
            for source in inferred_sources[1:]:
                if self.people[source].last_infection.date > inferred_infection.date:
                    inferred_infection = self.people[source].last_infection

        outbreak_id = inferred_infection.outbreak_id
        new_infection = self.__add_new_infection(who_name, when, outbreak_id)
        self.__increase_outbreak(when, outbreak_id)

        # Checking if the infection is direct
        if new_infection.who in self.connections[inferred_infection.who][:][1]:
            inferred_infection.add_direct_infection()
            self.direct_infections_connections[inferred_infection.who].add(who_name)

        return new_infection.infection_id

    def number_of_infected_clusters(self) -> int:
        """ Method to find the number of circles infected
        based on direct infections contained in direct_infections_connections.
        An infected circle is interpreted as a strongly consistent component that is not a
        singleton. The method is a modified Tarjan algorithm based on the given sources:
        'strongly connected components' chapter from the book Introduction to Algorithms;
        https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/

        Parameters:
            index -> used to give unique numbers to the vertices
            stack -> list of vertices that are part of the strongly
                     connected component currently being processed
            indices -> a dictionary whose keys are consecutive persons
                       and whose values are a three-element list.
                      The first item contains the index of the vertex being visited. If
                      vertex has not yet been visited then the value is -1. The second item
                      contains the lowest index of the vertex that can be reached by any other
                      vertex can reach. The last position contains information about
                      whether a vertex is currently on the stack.
        Returns:
            strongly_connected_components_count -> number of infected vertebrae

        Time complexity: O(V + F)
        Memory complexity: O(V)
        """
        strongly_connected_components_count = 0
        index = 0
        stack = []
        indices = {person: [-1, -1, False] for person in self.people_names}

        def DFSscc(node):
            """ Method that searches deep into the graph direct_infections_connections
            calling itself recursively to label each vertex.
            Since cycles can occur in a graph as opposed to a tree,
            we mark the visited vertices, in search of the vertex
            with the smallest index that is reachable for each vertex
            forming a given strong coherent component.
            """
            nonlocal strongly_connected_components_count
            nonlocal index
            indices[node] = [index, index, True]
            index += 1
            stack.append(node)

            for neighbor in self.direct_infections_connections[node]:
                if indices[neighbor][0] == -1:
                    DFSscc(neighbor)
                    indices[node][1] = min(indices[node][1], indices[neighbor][1]) # unvisited edge
                elif indices[neighbor][2]:
                    indices[node][1] = min(indices[node][1], indices[neighbor][0]) # back edge

            # Finding a strong coherent component
            if indices[node][1] == indices[node][0]:
                strongly_connected_component = 0
                connected_person = None
                while connected_person != node:
                    connected_person = stack.pop()
                    indices[connected_person][2] = False
                    strongly_connected_component += 1

                # The circle of infected cannot be a singleton
                if strongly_connected_component > 1:
                    strongly_connected_components_count += 1

        for person in self.people_names:
            if indices[person][0] == -1:
                DFSscc(person)

        return strongly_connected_components_count
