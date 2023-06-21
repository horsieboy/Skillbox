import joblib
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Функция выделения целевых действий
def prepare_target_df():
    print('Start preparing target df...')
    # Загрузка исходных датасетов
    df_hits = pd.read_csv('data/ga_hits-001.csv', low_memory=False)
    df_session = pd.read_csv('data/ga_sessions.csv', low_memory=False)

    # Выделение целевых переменных
    target = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
              'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
              'sub_submit_success', 'sub_car_request_submit_click']
    df_target = df_hits[df_hits['event_action'].isin(target)]
    df_session['event'] = df_session['session_id'].isin(df_target['session_id']).astype(int)

    # Откидываем лишние столбцы
    df_session = df_session.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number'])

    print('Done preparing target df')
    return df_session


# Преобразование типов
def prepare_types(df):
    print('Start preparing data types...')
    feature_to_str = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category',
                      'device_os', 'device_brand', 'device_model', 'device_screen_resolution',
                      'device_browser', 'geo_country', 'geo_city']
    for x in feature_to_str:
        df[x] = df[x].dropna().astype('str')
    print('Done preparing data types')
    return df


# Обработка пропусков
def fill_nan(df):
    print('Start filling NaN...')
    # Пропуски в device_model
    df['device_model'] = df['device_model'].fillna('noname')

    # Пропуски в utm_keyword
    df['utm_keyword'] = df['utm_keyword'].fillna('other')

    # Пропуски в device_brand
    df.loc[(df['device_category'] == 'desktop') & df['device_brand'].isna() & df['device_os'].isna(),
           ['device_os', 'device_brand']] = 'other'

    df.loc[(df['device_category'] == 'desktop') & df['device_brand'].isna() &
           ((df['device_os'] == 'Windows') |
            (df['device_os'] == 'Linux') |
            (df['device_os'] == 'Chrome OS') |
            (df['device_os'] == '(not set)')), 'device_brand'] = 'PC'

    df.loc[df['device_os'] == 'Macintosh', 'device_brand'] = 'Apple'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'desktop') & df['device_os'].isna(),
           'device_os'] = 'Macintosh'

    df.loc[(df['device_category'] == 'desktop') & df['device_os'].isna(), 'device_os'] = 'Linux'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'desktop') & df['device_os'].isna(),
           'device_os'] = 'iOS'

    df.loc[(df['device_category'] == 'mobile') & df['device_os'].isna(), 'device_os'] = 'Android'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'tablet') & df['device_os'].isna(),
           'device_os'] = 'iOS'

    df.loc[df['device_os'].isna() & (df['device_category'] == 'tablet'), 'device_os'] = 'Android'

    # Пропуски в utm_adcontent
    df['utm_adcontent'] = df['utm_adcontent'].fillna('other')

    # Пропуски в utm_campaign
    df['utm_campaign'] = df['utm_campaign'].fillna('other')

    # Остальные пропуски
    df = df.dropna()
    print('Done filling NaN')
    return df


# Кодирование категориальных переменных
def onehotencoding(df):
    print('Start onehotencoding...')
    columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
               'utm_keyword', 'device_category', 'device_os', 'device_brand',
               'device_model', 'device_screen_resolution', 'device_browser',
               'geo_country', 'geo_city']

    # Сделаем составной датасет где количество положительных событий не сильно отличалось от количества нецелевых
    n_event = len(df[df['event'] == 1])
    df_short = pd.concat([df[df['event'] == 1],
                          df[df['event'] == 0].sample(n=2*n_event, random_state=12)]).reset_index(drop=True)

    # Закодируем категориальные переменные
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, max_categories=100)
    ohe.fit(df_short.drop(columns='event')[columns])
    ohe_columns = ohe.transform(df_short[columns])
    df_prepared = pd.concat([pd.DataFrame(ohe_columns, columns=ohe.get_feature_names_out()), df_short['event']], axis=1)

    # Сохраним ohe в файл
    filename = 'data/ohe.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(ohe, file)

    print('Done onehotencoding')
    return df_prepared


def modeling(df_modeling):
    print('Start modeling...')

    # Выделим фичи и целевую переменную
    x = df_modeling.drop(['event'], axis=1)
    y = df_modeling['event']

    # Разделим на train и test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=12)

    # Инициализируем список классификаторов
    clfs = []

    # Подбор параметров модели логистической регрессии
    parameters_logreg = {'C': [0.25, 0.5, 1, 2]}
    logreg = LogisticRegression(solver='liblinear',
                                class_weight='balanced',
                                tol=1e-6,
                                random_state=12)
    clf_logreg = GridSearchCV(logreg, parameters_logreg, cv=4, scoring='roc_auc', verbose=100)
    clf_logreg.fit(x_train, y_train)
    clfs.append(clf_logreg)

    # Подбор параметров модели случайного леса
    parameters_rf = {'min_samples_split': [2, 3, 4]}
    rf = RandomForestClassifier(n_estimators=100,
                                max_features='sqrt',
                                min_samples_leaf=2,
                                bootstrap=False,
                                max_depth=100,
                                n_jobs=-1,
                                random_state=12)
    clf_rf = GridSearchCV(rf, parameters_rf, cv=4, scoring='roc_auc', verbose=100)
    clf_rf.fit(x_train, y_train)
    clfs.append(clf_rf)

    # Подбор параметров модели многослойного персептрона
    parameters_mlp = {'hidden_layer_sizes': [(2, 2), (5, 2)]}
    mlp = MLPClassifier(activation='identity',
                        solver='lbfgs',
                        alpha=0.0001,
                        tol=1e-3,
                        max_iter=1000,
                        random_state=12)
    clf_mlp = GridSearchCV(mlp, parameters_mlp, cv=4, scoring='roc_auc', verbose=100)
    clf_mlp.fit(x_train, y_train)
    clfs.append(clf_mlp)

    # Поиск наилучшего классификатора
    max_auc = 0
    for clf in clfs:
        if max_auc < clf.best_score_:
            max_auc = clf.best_score_
            best_clf = clf
    best_clf.fit(x, y)
    print(f'Done modeling \nmax_auc: {max_auc} \n{best_clf}')
    return best_clf


def main():
    model = modeling(onehotencoding(fill_nan(prepare_target_df())))
    filename = 'data/model.pkl'
    joblib.dump({
        'model': model.best_estimator_,
        'metadata': {
            'name': 'Sberavtopodpiska prediction model',
            'author': 'Vladimir Bocharov',
            'version': 1,
            'date': datetime.now(),
            'type': type(model.best_estimator_).__name__,
            'auc': model.best_score_
        }
    }, filename)


if __name__ == '__main__':
    main()
