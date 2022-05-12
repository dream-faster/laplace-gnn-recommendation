from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    num_users: int = 3
    num_user_features: int = 2
    num_articles: int = 10
    num_article_features: int = 5
    connection_ratio: float = 0.5
