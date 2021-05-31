from sklearn.model_selection import train_test_split

from tools import *
import argparse

# from word2number import w2n

def main(is_final_estimator=True):
    # Parameters
    seed = np.random.seed(0)

    if is_final_estimator:
        df_id = pd.read_csv('dataset/generated/COVID-chat-testdata.csv')
    else:
        df_id = pd.read_csv('dataset/generated/COVID-chat-trainingdata.csv')
    df_predicitons = pd.read_csv('dataset/generated/submission/infected.csv')

    df = pd.concat([df_id['ID'], df_predicitons['GPT2_Prediction']], axis=1)
    df.rename(columns={'GPT2_Prediction': 'Prediction'}, inplace=True)

    if is_final_estimator:
        df.to_csv('dataset/generated/submission/predicton.csv', index=False)
    else:
        df.to_csv('dataset/generated/transformed_train.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_final_estimator', action='store_true')

    args = parser.parse_args()
    main(args.is_final_estimator)