from neo4j import GraphDatabase


class HelloWorldExample:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.write_transaction(self._create_nodes, num_limit=2)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run(
            "CREATE (a:Greeting) "
            "SET a.message = $message "
            "RETURN a.message + ', from node ' + id(a)",
            message=message,
        )
        return result.single()[0]

    @staticmethod
    def _create_nodes(tx, num_limit):
        queries = []
        for name in ["Alice", "Bob", "Carol"]:
            queries.append(
                f"CREATE (person:Person) SET person.name = {name} RETURN person limit $num_limit"
            )

        result = tx.run(
            "".join(queries),
            num_limit=num_limit,
        )
        return result.single()[0]


if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://localhost:7687", "neo4j", "123456")
    greeter.print_greeting("hello, world")
    greeter.close()
