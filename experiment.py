from data.neo4j.neo4j_database import Database


db = Database("bolt://localhost:7687", "neo4j", "123456")

result = db.find_node(1, "Customer")

print(result)
