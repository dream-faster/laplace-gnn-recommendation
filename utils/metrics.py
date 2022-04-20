import torch
import numpy as np


def recall_at_k(
    all_ratings, k: int, num_customers: int, ground_truth, unique_customers, data_mp
) -> dict:
    """
    Calculates recall@k during validation/testing for a single batch.
    args:
      all_ratings: array of shape [number of customers in batch, number of articles in whole dataset]
      k: the value of k to use for recall@k
      num_customers: the number of customers in the dataset
      ground_truth: array of shape [2, X] where each column is a pair of (customer_idx, positive article idx). This is the
         batch that we are calculating metrics on.
      unique_customers: 1D vector of length [number of customers in batch], which specifies which customer corresponds
         to each row of all_ratings
      data_mp: an array of shape [2, Y]. This is all of the known message-passing edges. We will use this to make sure we
         don't recommend articles that are already known to be in the customer.
    returns:
      Dictionary of customer ID -> recall@k on that customer
    """
    # We don't want to recommend articles that are already known to be in the customer.
    # Set those to a low rating so they won't be recommended
    known_edges = data_mp[
        :, data_mp[0, :] < num_customers
    ]  # removing duplicate edges (since data_mp is undirected). also makes it so
    # that for each column, customer idx is in row 0 and article idx is in row 1
    customer_to_idx_in_batch = {
        customer: i for i, customer in enumerate(unique_customers.tolist())
    }
    exclude_customers, exclude_articles = (
        [],
        [],
    )  # already-known customer/article links. Don't want to recommend these again
    for i in range(known_edges.shape[1]):  # looping over all known edges
        pl, article = known_edges[:, i].tolist()
        if (
            pl in customer_to_idx_in_batch
        ):  # don't need the edges in data_mp that are from customers that are not in this batch
            exclude_customers.append(customer_to_idx_in_batch[pl])
            exclude_articles.append(
                article - num_customers
            )  # subtract num_customers to get indexing into all_ratings correct
    all_ratings[
        exclude_customers, exclude_articles
    ] = -10000  # setting to a very low score so they won't be recommended

    # Get top k recommendations for each customer
    _, top_k = torch.topk(all_ratings, k=k, dim=1)
    top_k += num_customers  # topk returned indices of articles in ratings, which doesn't include customers.
    # Need to shift up by num_customers to get the actual article indices

    # Calculate recall@k
    ret = {}
    for i, customer in enumerate(unique_customers):
        pos_articles = ground_truth[1, ground_truth[0, :] == customer]

        k_recs = top_k[i, :]  # top k recommendations for customer
        recall = len(np.intersect1d(pos_articles, k_recs)) / len(pos_articles)
        ret[customer] = recall
    return ret


def bpr_loss(
    users_emb_final,
    users_emb_0,
    pos_items_emb_final,
    pos_items_emb_0,
    neg_items_emb_final,
    neg_items_emb_0,
    lambda_val,
):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (
        users_emb_0.norm(2).pow(2)
        + pos_items_emb_0.norm(2).pow(2)
        + neg_items_emb_0.norm(2).pow(2)
    )  # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

    return loss


# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items


# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(
        r, dim=-1
    )  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor(
        [len(groundTruth[i]) for i in range(len(groundTruth))]
    )
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1.0 / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1.0 / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return torch.mean(ndcg).item()


# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num articles
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype("float"))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg
