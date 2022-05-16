from data.neo4j.neo4j import App


db = App(uri="bolt://localhost:7687", user="neo4j", password="password")
db.clear()
db.create_indexes()
db.load_articles_csv()
db.load_customers_csv()
db.load_relationships()
db.close()
