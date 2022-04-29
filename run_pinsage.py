# from pinsage.model import run_pinsage
from pinsage.process_hm import process_hm
from run_preprocessing import preprocess
from data.types import PreprocessingConfig, UserColumn, ArticleColumn, DataType

# preprocess(
#     PreprocessingConfig(
#         customer_features=[
#             UserColumn.PostalCode,
#             UserColumn.FN,
#             UserColumn.Age,
#             UserColumn.ClubMemberStatus,
#             UserColumn.FashionNewsFrequency,
#             UserColumn.Active,
#         ],
#         # customer_nodes=[],
#         article_features=[
#             ArticleColumn.ProductCode,
#             ArticleColumn.ProductTypeNo,
#             ArticleColumn.GraphicalAppearanceNo,
#             ArticleColumn.ColourGroupCode,
#         ],
#         # article_nodes=[],
#         article_non_categorical_features=[ArticleColumn.ImgEmbedding],
#         filter_out_unconnected_nodes=True,
#         load_image_embedding=False,
#         load_text_embedding=False,
#         text_embedding_colname="derived_look",
#         K=0,
#         data_size=None,
#         save_to_csv=False,
#         data_type=DataType.dgl,
#     )
# )
process_hm()
# run_pinsage()
