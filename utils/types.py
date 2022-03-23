from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    val_split: float
    test_split: float


@dataclass
class PreprocessingConfig:
    customer_features: list[UserColumn]
    customer_nodes: list[UserColumn]

    article_features: list[ArticleColumn]
    article_nodes: list[ArticleColumn]

    K: int
    data_size: int
