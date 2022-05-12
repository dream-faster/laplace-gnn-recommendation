from dataclasses import dataclass
import torch as t


@dataclass
class GeneratorConfig:
    num_users: int = 3
    num_user_features: int = 2
    num_articles: int = 10
    num_article_features: int = 5
    connection_ratio: float = 0.5


generator_config = GeneratorConfig(
    num_users=t.randint(low=1, high=10, size=(1,))[0].item(),
    num_user_features=t.randint(low=1, high=5, size=(1,))[0].item(),
    num_articles=t.randint(low=1, high=20, size=(1,))[0].item(),
    num_article_features=t.randint(low=1, high=8, size=(1,))[0].item(),
    connection_ratio=0.5,
)
