import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score

def preprocess(filepath, test_size=0.2, random_state=42):
    df = pd.read_json(filepath, lines=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # data partition
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # encode user_id and asin
    df_train['user_id_encoded'] = LabelEncoder().fit_transform(df_train['user_id'])
    df_train['asin_encoded'] = LabelEncoder().fit_transform(df_train['asin'])

    df_test['user_id_encoded'] = LabelEncoder().fit_transform(df_test['user_id'])
    df_test['asin_encoded'] = LabelEncoder().fit_transform(df_test['asin'])

    return df_train, df_test

def retrieval_train(df_train, epochs=50, num_threads=2, item_alpha=1e-6, user_alpha=1e-6):
    # create LightFM dataset
    dataset = Dataset()
    dataset.fit(df_train['user_id_encoded'], df_train['asin_encoded'])

    # interactions matrix
    interactions, _ = dataset.build_interactions(
        (row['user_id_encoded'], row['asin_encoded']) for _, row in df_train.iterrows()
    )

    model = LightFM(loss='warp', item_alpha=item_alpha, user_alpha=user_alpha)

    # train model
    model.fit(interactions, epochs=epochs, num_threads=num_threads)

    return model, dataset

def evaluate_model(model, dataset, df, num_threads=2):
    interactions, _ = dataset.build_interactions(
        (row['user_id_encoded'], row['asin_encoded']) for _, row in df.iterrows()
    )

    precision = precision_at_k(model, interactions, k=5, num_threads=num_threads).mean()
    auc = auc_score(model, interactions, num_threads=num_threads).mean()

    return precision, auc

def get_top_rec(model, user_id, asin, n=100):
    scores = model.predict(user_id, asin)
    top_items = np.argsort(-scores)[:n]
    return top_items

def generate_recommendations(model, df_test, n=100):
    user_ids = df_test['user_id_encoded'].unique()
    all_items = df_test['asin_encoded'].unique()

    recommendations = {}
    for user_id in user_ids:
        recommendations[user_id] = get_top_rec(model, user_id, all_items, n=n)
    
    return recommendations

if __name__ == "__main__":
    filepath = './data/Gift_Cards.jsonl'

    # preprocess the data
    df_train, df_test = preprocess(filepath)

    # train the retrieval model
    model, dataset = retrieval_train(df_train)

    # evaluate the model
    train_precision, train_auc = evaluate_model(model, dataset, df_train)
    test_precision, test_auc = evaluate_model(model, dataset, df_test)

    print(f'Train precision: {train_precision:.2f}')
    print(f'Train AUC: {train_auc:.2f}')
    print(f'Test precision: {test_precision:.2f}')
    print(f'Test AUC: {test_auc:.2f}')

    # generate top 100 recommendations for each user
    # recommendations = generate_recommendations(model, df_test)
