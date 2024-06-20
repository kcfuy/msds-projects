import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import pickle
from itertools import product
from surprise import Reader, Dataset, SVD, NormalPredictor
from surprise import accuracy
from surprise.model_selection import cross_validate
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def create_sqlite_db():
    """Create sqlite3 db from csv files"""
    conn = sqlite3.connect('BoardGames.db')
    fp = '/mnt/data/public/bgg/'
    pd.read_csv(fp + 'bgg-19m-reviews.csv',
                index_col=0)[
        ['user', 'ID', 'rating']
    ].to_sql('reviews', conn, if_exists='replace', index=False)
    pd.read_csv(
        fp + 'games_detailed_info.csv',
        index_col=0,
        low_memory=False
    ).to_sql('games_info', conn, index=False, if_exists='replace')
    return conn


def create_pickle_files(conn):
    """
    Create pickle files for game lists and profiles from the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        A connection object to the SQLite database.

    Returns
    -------
    None
    """
    games_list = pd.read_sql(
        """
        SELECT
            ID,
            COUNT(user) as count
        FROM
            reviews
        GROUP BY 1
        HAVING count >= 100
        ORDER BY 2 DESC
        LIMIT 10000
        """,
        conn
    )['ID'].tolist()
    raw_games_df = pd.read_sql(
        f"""
        SELECT
            id,
            "primary",
            boardgamecategory
        FROM
            games_info g
        WHERE 1=1
        AND boardgamecategory is not null
        AND boardgamemechanic is not null
        AND id in {tuple(games_list)}
        """,
        conn
    )
    games_df = raw_games_df.drop('primary', axis=1).copy()
    pattern = r'["\']([^"\']+)["\']'
    categ = games_df['boardgamecategory'].str.extractall(pattern).reset_index()
    games_df['category'] = categ.groupby('level_0')[0].apply(
        ', '.join
    ).str.split(', ')
    mlb = MultiLabelBinarizer()
    df_encoded = mlb.fit_transform(games_df['category'])
    df_encoded = pd.DataFrame(df_encoded,
                              columns=mlb.classes_,
                              index=games_df['id'])
    games_df = games_df.set_index('id')
    df_item_profile = pd.concat([games_df, df_encoded.iloc[:, 1:]], axis=1)
    df_item_profile = df_item_profile.drop(
        columns=['boardgamecategory', 'category']
    )
    raw_games_df = raw_games_df.set_index('id')
    df_raw_item_profile = pd.concat(
        [raw_games_df, df_encoded.iloc[:, 1:]],
        axis=1
    )
    df_raw_item_profile = df_raw_item_profile.drop(
        columns=['boardgamecategory']
    )
    final_games_list = df_item_profile.index.tolist()
    user_list = pd.read_sql(
        """
        SELECT
            user,
            count(rating) as count_r
        FROM
            reviews
        GROUP BY
            1
        HAVING count_r >= 50
        ORDER BY
            2 DESC
        LIMIT 10000
        """,
        conn
    )['user'].tolist()
    df_reviews = pd.read_sql(
        f"""
        SELECT
            user,
            ID as item_id,
            rating
        FROM
            reviews
        WHERE 1=1
        AND user in {tuple(user_list)}
        AND ID in {tuple(final_games_list)}
        ORDER BY
            1
        """,
        conn
    )
    df_reviews.to_pickle('reviews.pkl')
    df_item_profile.to_pickle('df_item_profile.pkl')
    df_raw_item_profile.to_pickle('df_raw_item.pkl')


def normal_pred(df_reviews):
    """
    Calculate normal predictor model performance metrics using cross-validation

    Parameters
    ----------
    df_reviews : pandas.DataFrame
        DataFrame containing user reviews.

    Returns
    -------
    dict
        A dictionary with cross-validation results.
    """
    df_sampled = df_reviews.sample(n=1_000_000, random_state=0)
    reader = Reader(rating_scale=(0,10))
    data = Dataset.load_from_df(df_sampled, reader)
    algo = NormalPredictor()
    cv_result = cross_validate(algo,
                               data,
                               measures=['rmse', 'mae'],
                               cv=5,
                               n_jobs=-1)
    return cv_result


def cross_validate_lf(df_reviews):
    """
    Calculate latent factor model performance metrics using cross-validation.

    Parameters
    ----------
    df_reviews : pandas.DataFrame
        DataFrame containing user reviews.

    Returns
    -------
    dict
        A dictionary with cross-validation results.
    """
    df_sampled = df_reviews.sample(n=1_000_000, random_state=0)
    reader = Reader(rating_scale=(0,10))
    data = Dataset.load_from_df(df_sampled, reader)
    algo = SVD(n_factors=100, random_state=0)
    cv_result = cross_validate(algo,
                               data,
                               measures=['rmse', 'mae'],
                               cv=5,
                               n_jobs=-1)
    return cv_result


def compute_user_profile_agg_unary(df_utility, df_item_profiles, user):
    """
    Compute the aggregated unary user profile.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    df_item_profiles : pandas.DataFrame
        Item profile data.
    user : str
        User identifier.

    Returns
    -------
    pandas.Series
        Aggregated unary user profile.
    """
    user_mean = df_utility.loc[user].mean()
    idx = np.where(df_utility.loc[user] >= user_mean)[0]
    user_profile = df_item_profiles.iloc[idx].mean()
    return user_profile


def filled_agg_unary(df_utility, df_item_profiles, user_profile, user):
    """
    Fill user ratings based on aggregated unary user profile.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    df_item_profiles : pandas.DataFrame
        Item profile data.
    user_profile : pandas.Series
        Aggregated unary user profile.
    user : str
        User identifier.

    Returns
    -------
    dict
        Dictionary of item recommendations and their scores.
    """
    nan_idx = np.isnan(df_utility.loc[user])
    items = df_item_profiles.loc[nan_idx]
    return {i: cosine(item, user_profile)
            for i, item in items.iterrows()
            if cosine(item, user_profile) > 0}

def apply_idf_scaling(df):
    """
    Apply IDF scaling to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to apply IDF scaling.

    Returns
    -------
    pandas.DataFrame
        DataFrame with IDF-scaled features.
    """

    N = df.shape[0]
    nt = (df > 0).sum(axis=0)
    nt = nt.replace(0, 1)
    idf = np.log(N / nt)
    scaled_df = df.multiply(idf, axis=1)
    return scaled_df


def remove_items(df_utility, user, seed, L):
    """
    Randomly remove L items from a user's profile in the utility matrix.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    user : str
        User identifier.
    seed : int
        Random seed for reproducibility.
    L : int
        Number of items to remove.

    Returns
    -------
    tuple
        A tuple containing the removed items and their original ratings.
    """
    removed_idx = {}
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, df_utility.shape[1])
        if np.isnan(df_utility.loc[user].iloc[idx]):
            pass
        else:
            removed_idx[idx] = df_utility.loc[user].iloc[idx]
            df_utility.loc[user].iloc[idx] = np.nan

        if len(removed_idx) >= L:
            break

    return removed_idx, sorted(removed_idx.items(), key=lambda x: (-x[1]))


def remove_items2(df_utility, user, seed, rated_items):
    """
    Randomly remove items from a user's rated list in the utility matrix.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    user : str
        User identifier.
    seed : int
        Random seed for reproducibility.
    rated_items : list
        List of rated items to be considered for removal.

    Returns
    -------
    tuple
        A tuple containing the removed items and their original ratings.
    """
    removed_idx = {}
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.integers(0, df_utility.shape[1])
        true_idx = df_utility.columns[idx]
        if true_idx in rated_items:
            removed_idx[true_idx] = df_utility.loc[user][true_idx]
            df_utility.loc[user][true_idx] = np.nan
        else:
            pass

        if len(removed_idx) >= 10:
            break

    return removed_idx, sorted(removed_idx.items(), key=lambda x: (-x[1]))


def get_unary_ndcg(df_utility, df_item_profile, user, removed_idx,
                   removed_ranked):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for unary ratings.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    df_item_profile : pandas.DataFrame
        Item profile data.
    user : str
        User identifier.
    removed_idx : dict
        Dictionary of removed item indices and their ratings.
    removed_ranked : list
        List of removed items and their ratings, ranked by original rating.

    Returns
    -------
    float
        NDCG score.
    """
    user_profile_agg_unary = compute_user_profile_agg_unary(
        df_utility, df_item_profile, user
    )
    ratings = filled_agg_unary(
        df_utility, df_item_profile, user_profile_agg_unary, user
    )
    pred = []
    for k, v in removed_ranked:
        true_idx = df_utility.columns[k]
        pred.append((k, ratings[true_idx]))
    sorted_pred = sorted(pred, key=lambda x: (-x[1]))
    sorted_pred_idx = [k for k, _ in sorted_pred]
    m = 1
    dcg = np.mean(
            [
                np.sum(
                    [(2 ** removed_idx[k] - 1) for k in sorted_pred_idx]
                    / np.log2(np.arange(len(sorted_pred_idx)) + 2)
                )
                for i in range(m)
            ]
        )
    idcg = np.mean(
            [
                np.sum(
                    [(2 ** k[1] - 1) for k in removed_ranked]
                    / np.log2(np.arange(len(sorted_pred_idx)) + 2)
                )
                for i in range(m)
            ]
        )
    ndcg = dcg / idcg
    return ndcg


def cv_content(df_utility, df_item_scaled, users, seeds, L):
    """
    Perform content-based filtering cross-validation.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility matrix with user-item interactions.
    df_item_scaled : pandas.DataFrame
        Item profile data with applied scaling.
    users : list
        List of user identifiers.
    seeds : list
        List of random seeds for reproducibility.
    L : int
        Number of items to remove for testing.

    Returns
    -------
    dict
        Dictionary of users and their NDCG scores.
    """
    unary_ndcg_dict = {}
    for user in users:
        unary_ndcg_dict[user] = []
        for seed in seeds:
            removed_idx, removed_ranked = remove_items(df_utility, user, seed,
                                                       L)
            unary_ndcg_dict[user].append(get_unary_ndcg(df_utility,
                                                        df_item_scaled,
                                                        user,
                                                        removed_idx,
                                                        removed_ranked))
    return unary_ndcg_dict


def get_ndcg_cb(df_reviews, df_item_profile, L):
    """
    Calculate NDCG for content-based filtering.

    Parameters
    ----------
    df_reviews : pandas.DataFrame
        DataFrame containing user reviews.
    df_item_profile : pandas.DataFrame
        Item profile data.
    L : int
        Number of items to consider in NDCG calculation.

    Returns
    -------
    float
        Average NDCG score across users.
    """
    df_utility = df_reviews.pivot(
        columns='item_id',
        index='user',
        values='rating'
    )
    df_item_scaled = apply_idf_scaling(df_item_profile)
    rng = np.random.default_rng(1)
    users_idx = rng.integers(0, df_utility.shape[0], 10)
    users = df_utility.index[users_idx]
    seeds = rng.integers(0, df_utility.shape[1], L)
    unary_ndcg_dict = cv_content(df_utility,
                                 df_item_scaled,
                                 users,
                                 seeds,
                                 L)
    return np.mean([np.mean(lst) for lst in unary_ndcg_dict.values()])


def lf_training(df_reviews):
    """
    Train a latent factor model using the provided reviews.

    Parameters
    ----------
    df_reviews : pandas.DataFrame
        DataFrame containing user reviews.

    Returns
    -------
    surprise.prediction_algorithms.matrix_factorization.SVD
        Trained SVD model.
    """
    reader = Reader(rating_scale=(0,10))
    data = Dataset.load_from_df(
        df_reviews,
        reader)
    algo = SVD(n_factors=100, random_state=0, verbose=False)
    algo_fitted = algo.fit(data.build_full_trainset())
    return algo_fitted


def get_user_predictions(user, algo, df):
    """
    Predict ratings for all items for a specific user.

    Parameters
    ----------
    user : str
        User identifier.
    algo : surprise.prediction_algorithms.algo_base.AlgoBase
        Trained prediction algorithm.
    df : pandas.DataFrame
        DataFrame containing user reviews.

    Returns
    -------
    dict
        Dictionary containing various prediction results.
    """
    items = df['item_id'].unique().tolist()
    results = {}
    rated_items = (df[df['user'] == user]['item_id']
                   .unique()
                   .tolist())
    user_ratings = df[df['user']==user]
    unrated_items = np.setdiff1d(items, rated_items, True)
    testset = [(u, i, r) for u, i, r in product([user],
                                                unrated_items,
                                                [user_ratings['rating'].mean()]
                                               )]
    predictions = algo.test(testset)

    res = pd.DataFrame(predictions).sort_values('est', ascending=False)
    res = res[['uid', 'iid', 'est']].rename(
        {
            'uid': 'user',
            'iid': 'item_id',
            'est': 'rating'
        },
        axis=1
    )
    user_long = pd.concat([user_ratings, res])
    user_utility = user_long.set_index('item_id').drop('user',
                                                       axis=1).squeeze()

    results['rated_items'] = rated_items
    results['unrated_items'] = unrated_items
    results['predictions'] = predictions
    results['user_utility'] = user_utility

    return results


def compute_user_profile(user_utility, df_item_profiles, rated_items):
    """
    Compute a user's profile based on their rated items.

    Parameters
    ----------
    user_utility : pandas.Series
        Utility data for a specific user.
    df_item_profiles : pandas.DataFrame
        Item profile data.
    rated_items : list
        List of items rated by the user.

    Returns
    -------
    pandas.Series
        Computed user profile.
    """
    user_mean = user_utility.loc[rated_items].mean()
    idx = np.where(user_utility.loc[rated_items] >= user_mean)[0]
    user_profile = df_item_profiles.iloc[idx].mean(axis=0)
    return user_profile


def get_user_top_L(user_utility, rated_items, L=10):
    """
    Get top L rated items for a user.

    Parameters
    ----------
    user_utility : pandas.Series
        Utility data for a specific user.
    rated_items : list
        List of items rated by the user.
    L : int, optional
        Number of top items to return, default is 10.

    Returns
    -------
    pandas.Series
        Top L rated items by the user.
    """
    items = user_utility.loc[rated_items].sort_values(ascending=False)[:L]
    return items


def recommend_L(df_item_profiles, user_profile, nan_idx, L=10):
    """
    Recommend top L items based on the user profile.

    Parameters
    ----------
    df_item_profiles : pandas.DataFrame
        Item profile data.
    user_profile : pandas.Series
        Computed user profile.
    nan_idx : list
        Indices of items to consider for recommendation.
    L : int, optional
        Number of recommendations to return, default is 10.

    Returns
    -------
    list
        List of recommended item indices.
    """
    items = df_item_profiles.loc[nan_idx]
    ratings = sorted([(i,
                       cosine_similarity(
                           item.to_numpy().reshape(1,-1),
                           user_profile.to_numpy().reshape(1,-1)
                       )
                      )
                      for i, item in items.iterrows()
                     ], key=lambda x: (-x[1], x[0]))
    return [i for i, _ in ratings[:L]]


def get_recommendations_users(results,
                              df_raw_item_profile,
                              df_item_profile,
                              sampled_items,
                              L=5):
    """
    Generate recommendations for multiple users.

    Parameters
    ----------
    results : dict
        Prediction results for each user.
    df_raw_item_profile : pandas.DataFrame
        Raw item profile data.
    df_item_profile : pandas.DataFrame
        Scaled item profile data.
    sampled_items : list
        List of item indices to consider.
    L : int, optional
        Number of top items to recommend, default is 5.

    Returns
    -------
    dict
        Recommendations and profiles for each user.
    """
    df_item_profile_sliced = df_item_profile.loc[sampled_items]
    df_item_scaled = apply_idf_scaling(df_item_profile_sliced)
    df_raw = df_raw_item_profile.loc[sampled_items]
    recommendations = {}
    for user, user_dict in results.items():
        recommendations[user] = {}
        user_profile_agg_unary = compute_user_profile(
            user_dict['user_utility'],
            df_item_scaled,
            user_dict['rated_items']
            )
        # print(user_profile_agg_unary.to_numpy())
        top_items = get_user_top_L(
            user_dict['user_utility'],
            user_dict['rated_items'],
            L=L
            )
        agg_unary_recos = recommend_L(
            df_item_scaled,
            user_profile_agg_unary,
            user_dict['unrated_items'],
            L=L
            )
        user_top_L = df_raw.loc[top_items.index]
        user_top_L['rating'] = top_items

        user_recos = df_raw_item_profile.loc[agg_unary_recos]
        user_recos['rating'] = user_dict['user_utility'].loc[agg_unary_recos]

        recommendations[user]['orig'] = user_top_L
        recommendations[user]['reco'] = user_recos
        recommendations[user]['profile'] = user_profile_agg_unary

    return recommendations


def get_hybrid_ndcg(df_sampled, df_item_profile, algo, L):
    """
    Calculate NDCG for a hybrid recommendation system.

    Parameters
    ----------
    df_sampled : pandas.DataFrame
        Sampled DataFrame containing user reviews.
    df_item_profile : pandas.DataFrame
        Item profile data.
    algo : surprise.prediction_algorithms.algo_base.AlgoBase
        Trained prediction algorithm.
    L : int
        Number of items to consider in NDCG calculation.

    Returns
    -------
    float
        NDCG score for the hybrid system.
    """
    df_utility = df_sampled.pivot(
        columns='item_id',
        index='user',
        values='rating'
    )
    df_item_scaled = apply_idf_scaling(df_item_profile)
    rng = np.random.default_rng(0)
    users_idx = rng.integers(0, df_utility.shape[0], 10)
    users = df_utility.index[users_idx]
    seeds = rng.integers(0, df_utility.shape[1], L)
    unary_ndcg_dict = {}
    results = {}
    for user in users:
        results[user] = get_user_predictions(user, algo, df_sampled)
        df_utility.loc[user] = results[user]['user_utility']
        unary_ndcg_dict[user] = []
        for seed in seeds:

            removed_idx = {}
            rng = np.random.default_rng(seed)
            while True:
                idx = rng.integers(0, df_utility.shape[1])
                true_idx = df_utility.columns[idx]
                if true_idx in results[user]['rated_items']:
                    removed_idx[true_idx] = (results[user]['user_utility']
                                             [true_idx])
                    df_utility.loc[user][true_idx] = np.nan
                else:
                    pass

                if len(removed_idx) >= 10:
                    break
            removed_ranked = sorted(removed_idx.items(), key=lambda x: (-x[1]))
            user_mean = df_utility.loc[user].mean()
            idx = np.where(df_utility.loc[user] >= user_mean)[0]
            user_profile = df_item_profile.iloc[idx].mean()
            nan_idx = list(removed_idx.keys())
            nan_idx.extend(results[user]['unrated_items'])
            items = df_item_profile.loc[nan_idx]
            ratings = {i: cosine(item, user_profile)
                          for i, item in items.iterrows()
                          if cosine(item, user_profile) > 0}
            # print(removed_ranked)
            pred = []
            for k, v in removed_ranked:
                pred.append((k, ratings[k]))
            sorted_pred = sorted(pred, key=lambda x: (-x[1]))
            sorted_pred_idx = [k for k, _ in sorted_pred]
            m = 1
            dcg = np.mean(
                    [
                        np.sum(
                            [(2 ** removed_idx[k] - 1)
                             for k in sorted_pred_idx]
                            / np.log2(np.arange(len(sorted_pred_idx)) + 2)
                        )
                        for i in range(m)
                    ]
                )
            idcg = np.mean(
                    [
                        np.sum(
                            [(2 ** k[1] - 1) for k in removed_ranked]
                            / np.log2(np.arange(len(sorted_pred_idx)) + 2)
                        )
                        for i in range(m)
                    ]
                )
            ndcg = dcg / idcg
            unary_ndcg_dict[user].append(ndcg)

    return np.mean([np.mean(lst) for lst in unary_ndcg_dict.values()])