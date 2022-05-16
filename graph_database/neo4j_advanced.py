import logging
import sys

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    @staticmethod
    def enable_log(level, output_stream):
        handler = logging.StreamHandler(output_stream)
        handler.setLevel(level)
        logging.getLogger("neo4j").addHandler(handler)
        logging.getLogger("neo4j").setLevel(level)

    def create_transaction(self, user_name, article_name):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.write_transaction(
                self._create_and_return_transaction,
                user_name,
                article_name,
            )
            for row in result:
                print(
                    "Created transaction between: {p} buys {a} ".format(
                        p=row["p"], a=row["a"]
                    )
                )

    def create_constraints(self):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            session.write_transaction(self._create_constraints)

    @staticmethod
    def _create_and_return_transaction(tx, user_id, article_id):
        # To learn more about the Cypher syntax, see https://neo4j.com/docs/cypher-manual/current/
        # The Reference Card is also a good resource for keywords https://neo4j.com/docs/cypher-refcard/current/
        # query = (
        #     "CREATE (p:Person { id: $user_id }) "
        #     "CREATE (a:Article { id: $article_id }) "
        #     "CREATE (p)-[k:BUYS]->(a) "
        #     "RETURN p, a"
        # )
        query = (
            "MERGE (p:Person { id: $user_id }) "
            "MERGE (a:Article { id: $article_id }) "
            "MERGE (p)-[k:BUYS]->(a) "
            "RETURN p, a"
        )
        result = tx.run(
            query,
            user_id=user_id,
            article_id=article_id,
        )
        try:
            return [
                {
                    "p": row["p"]["id"],
                    "a": row["a"]["id"],
                }
                for row in result
            ]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error(
                "{query} raised an error: \n {exception}".format(
                    query=query, exception=exception
                )
            )
            raise

    def find_person(self, person_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_and_return_person, person_name)
            for row in result:
                print("Found person: {row}".format(row=row))

    @staticmethod
    def _find_and_return_person(tx, person_name):
        query = (
            "MATCH (p:Person) " "WHERE p.name = $person_name " "RETURN p.name AS name"
        )
        result = tx.run(query, person_name=person_name)
        return [row["name"] for row in result]

    @staticmethod
    def _create_constraints(tx):
        tx.run(
            "CREATE CONSTRAINT unique_user_id IF NOT EXISTS FOR (person:Person) REQUIRE person.id IS UNIQUE"
        )
        tx.run(
            "CREATE CONSTRAINT unique_article_id IF NOT EXISTS FOR (article:Article) REQUIRE article.id IS UNIQUE"
        )


if __name__ == "__main__":
    bolt_url = "bolt://localhost:7687"
    user = "neo4j"
    password = "123456"
    App.enable_log(logging.INFO, sys.stdout)
    app = App(bolt_url, user, password)
    app.create_transaction("User 3", "Article 2")
    app.create_constraints()
    app.find_person("Alice")
    app.close()
