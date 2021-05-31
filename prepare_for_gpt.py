from sklearn.model_selection import train_test_split

from tools import *

# import shapefile
# from shapely.geometry import Point # Point class
# from shapely.geometry import shape # shape() is a function to convert geo objects through the interface
# import countries.countries
# from word2number import w2n

def unzip_coordinates(df):
    # device location
    pattern = r'^Device location : \[(-?[1-9]?[0-9]?(?:\.[0-9]+)?), -?[1-9]?[0-9]{0,2}(?:\.[0-9]+)?\]$'
    df['latitude'] = df['location'].str.extract(pattern)
    df['latitude'] = pd.to_numeric(df['latitude'])

    pattern = r'^Device location : \[-?[1-9]?[0-9]?(?:\.[0-9]+), (-?[1-9]?[0-9]{0,2}(?:\.[0-9]+))\]$'
    df['longitude'] = df['location'].str.extract(pattern)
    df['longitude'] = pd.to_numeric(df['longitude'])

    df.longitude.fillna(df.longitude.median(), inplace=True)
    df.latitude.fillna(df.latitude.median(), inplace=True)

    
    # h_infected, xedges, yedges = np.histogram2d(df.longitude[df.Prediction == 1], df.latitude[df.Prediction == 1], bins=bins)
    # h_total, xedges, yedges = np.histogram2d(df.longitude, df.latitude, bins=bins)
    # p_infected = h_infected / h_total
    # # p_infected is 2-d array [x, y] with shape (14, 14)
    # # xedges denotes N+1 edges of bins from left to right (x1-bin1-x2-bin2-...-binN-xN+1)
    
    return df


def main():
    # Parameters
    seed = np.random.seed(0)
    target_feature = 'Prediction'

    # process train data

    df = load_data(labels_path='dataset/train.csv', conversations_folder_path='dataset/trainConversations', verbose=1)
    # drop duplicates, empty rows and columns and rows with invalid labels
    df.dropna(axis=0, how='any', subset=[target_feature], inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.drop_duplicates(inplace=True)
    
    df.to_csv('dataset/generated/COVID-chat-trainingdata.csv', index=False)


    df2 = load_data(labels_path='dataset/test.csv', conversations_folder_path='dataset/testConversations', verbose=1)
    df2.to_csv('dataset/generated/COVID-chat-testdata.csv', index=False)

if __name__ == '__main__':
    main()