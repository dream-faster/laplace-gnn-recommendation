import torch
import pandas as pd
from sklearn import preprocessing
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData, InMemoryDataset, download_url
from torch_geometric.utils import negative_sampling


class HMDataset(InMemoryDataset):
    image_embeddings_url = "https://storage.googleapis.com/heii-public/fashion-recommendation-image-embeddings-clip-ViT-B-32.pt"
    text_embeddings_url = "https://storage.googleapis.com/heii-public/fashion-recommendation-text-embeddings-clip-ViT-B-32.pt"
    raw_dir = "./data/original"
    processed_dir = "./data/derived"

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "fashion-recommendation-image-embeddings-clip-ViT-B-32.pt",
            "fashion-recommendation-text-embeddings-clip-ViT-B-32.pt",
            "articles.csv",
            "customers.csv",
            "transactions_train.csv",
        ]

    @property
    def processed_file_names(self):
        return f"hm_graph.pt"

    def download(self):
        download_url(self.image_embeddings_url, self.raw_dir)
        download_url(self.text_embeddings_url, self.raw_dir)

    def process(self):
        self.articles = pd.read_csv(self.raw_paths[2], index_col="article_id")
        self.customers = pd.read_csv(self.raw_paths[3], index_col="customer_id").fillna(
            0.0
        )
        self.transactions = pd.read_csv(self.raw_paths[4])

        data = HeteroData()
        self.article_image_embeddings = torch.load(self.raw_paths[0])
        self.article_text_embeddings = torch.load(self.raw_paths[1])

        # encode customers
        le = preprocessing.LabelEncoder()
        self.customers["postal_code"] = le.fit_transform(self.customers["postal_code"])
        self.customers.loc[
            self.customers["fashion_news_frequency"] == "None", "fashion_news_frequency"
        ] = 0.0
        self.customers.loc[
            self.customers["fashion_news_frequency"] == "NONE", "fashion_news_frequency"
        ] = 0.0
        customer_features = self.customers[
            [
                "postal_code",
                "age",
                "fashion_news_frequency",
                "FN",
                "Active",
                "club_member_status",
            ]
        ]
        customer_features = pd.get_dummies(
            customer_features,
            columns=["age", "fashion_news_frequency", "club_member_status"],
        )
        customer_features = torch.from_numpy(customer_features.to_numpy()).float()

        # encode articles
        self.articles = self.articles.merge(
            self.transactions.groupby("article_id")["price"].mean(),
            on="article_id",
            how="outer",
        ).fillna(0.0)
        # self.articles["price_bin"] = pd.qcut(self.articles["price"], 100, labels=False)
        self.articles["product_type_no"] = self.articles["product_type_no"].astype(str)
        product_type_no_le = preprocessing.LabelEncoder()
        self.articles["product_type_no"] = product_type_no_le.fit_transform(
            self.articles["product_type_no"]
        )
        self.articles["graphical_appearance_no"] = self.articles[
            "graphical_appearance_no"
        ].astype(str)
        graphical_appearance_no_le = preprocessing.LabelEncoder()
        self.articles[
            "graphical_appearance_no"
        ] = graphical_appearance_no_le.fit_transform(
            self.articles["graphical_appearance_no"]
        )
        article_features = self.articles[
            ["product_type_no", "graphical_appearance_no", "price"]
        ]
        # article_features = pd.get_dummies(
        #     article_features,
        #     columns=["price_bin"],
        # )
        article_features = torch.from_numpy(article_features.to_numpy()).float()
        article_features = torch.cat(
            (
                article_features,
                torch.stack(
                    self.articles.apply(
                        lambda article: self.article_image_embeddings.get(
                            int(article.name), torch.zeros(512)
                        ),
                        axis=1,
                    ).tolist()
                ),
            ),
            1,
        )
        for key in ["derived_name", "derived_look", "derived_category"]:
            article_features = torch.cat(
                (
                    article_features,
                    torch.stack(
                        self.articles.apply(
                            lambda article: self.article_text_embeddings[
                                int(article.name)
                            ].get(key, torch.zeros(512)),
                            axis=1,
                        ).tolist()
                    ),
                ),
                1,
            )

        # create nodes
        data["article"].x = article_features
        data["customer"].x = customer_features

        # create node edges
        t = self.transactions.to_dict()
        customers_id_ix = {v: k for k, v in enumerate(self.customers.index.unique())}
        # customers_ix_id = {k: v for k, v in enumerate(self.customers.index.unique())}
        articles_id_ix = {v: k for k, v in enumerate(self.articles.index.unique())}
        # articles_ix_id = {k: v for k, v in enumerate(self.articles.index.unique())}
        src = [customers_id_ix[t["customer_id"][i]] for i in t["customer_id"]]
        dst = [articles_id_ix[t["article_id"][i]] for i in t["article_id"]]
        data["customer", "buys", "article"].edge_index = torch.tensor(
            [src, dst]
        ).float()

        # transform?
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # PyTorch tensor functionality:
        # data = data.pin_memory()
        # data = data.to('cuda:0', non_blocking=True)
        torch.save(self.collate([data]), self.processed_paths[0])


if __name__ == "__main__":
    dataset = HMDataset("./data")
