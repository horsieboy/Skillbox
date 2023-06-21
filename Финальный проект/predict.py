import dill
import joblib
import pandas as pd


def transform_data(data):
    with open('data/ohe.pkl', 'rb') as file:
        ohe = dill.load(file)
    ohe_columns = ohe.transform(data)
    df_prepared = pd.DataFrame(ohe_columns, columns=list(ohe.get_feature_names_out()))
    return df_prepared


def load_data():
    data = pd.read_csv('data/df_sample.csv')
    return data


def predict_on_data(data):
    model = joblib.load('data/model.pkl')
    return pd.DataFrame(model['model'].predict(data), columns=['event'])


def main():
    df = transform_data(load_data())
    predict = predict_on_data(df)
    print(predict)
    print(predict.value_counts())
    predict.to_csv('data/predict_sample.csv', index_label=False, sep=',')


if __name__ == '__main__':
    main()
