from pandas.core import indexing
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.utils import _to_object_array

# Requires the PyMongo package.
# https://api.mongodb.com/python/current

PATH_MONGODB = 'mongodb://bigdata-1:27017' #/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'

# ====== Funcao para transformar os dicionarios do mongo db em DataFrame ======
def dict_pandas(dicts):
    dict_mongo={}
    a = 0
    for doc in dicts:
        if a == 0:
            dict_mongo = doc
            for u, v in doc.items():
                dict_mongo[u] = [v]
        else:
            for u, v in doc.items():
                dict_mongo[u].append(v)
        a = 1
    del dict_mongo['_id']

    return pd.DataFrame(dict_mongo)

if __name__ == '__main__':
    # conexao com o mongodb localizado na virtual Machine
    client = MongoClient(PATH_MONGODB)
    # Instanciando o database bigdata para a variavel db
    db = client['bigdata']

    # ====== ETL DO ARQUIVO CSV PARA O MONGODB ======
    df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv', delimiter = ',')
    # criando colecao
    db.create_collection('owid-covid-data')
    # inserindo dados na colecao criada
    db['owid-covid-data'].insert_many(dict(row[1]) for row in df.iterrows())
    

    # ==== EXTRACAO E TRATATAMENTO DOS DADOS VIA MONGO DB ====
    # dados a serem filtrados
    findfilter = {'iso_code':'BRA'}
    # colunas retornadas
    colunas = {'date': 1,
        'people_fully_vaccinated': 1,
        'population': 1
        }

    # unindo dicionarios extraidos atraves do mongodb
    
    result = db['owid-covid-data'].find(filter, colunas)

    # criando um dataframe atraves da funcao dict_pandas
    df_brasil = dict_pandas(result)
    
    #df_brasil.info()
    # convertendo campo para data
    df_brasil['date'] = pd.to_datetime(df_brasil['date'])
    # ordenando valores por data
    df_brasil = df_brasil.sort_values(by = 'date', ignore_index=True)
    
    # removendo para valores que nao foram computados dados da vacinacao
    index_nan = df_brasil.index[df_brasil['people_fully_vaccinated'].isnull()].tolist()
    
    for i in index_nan:
        if i<=384:
            continue
        elif pd.isnull(df_brasil['people_fully_vaccinated'].iloc[index_nan[len(index_nan) - 1]]):
            df_brasil['people_fully_vaccinated'].iloc[i] = df_brasil['people_fully_vaccinated'].iloc[i-1]
        elif not pd.isnull(df_brasil['people_fully_vaccinated'].iloc[i-1]):
            df_brasil['people_fully_vaccinated'].iloc[i] = df_brasil['people_fully_vaccinated'].iloc[i-1]
        elif not pd.isnull(df_brasil['people_fully_vaccinated'].iloc[i-2]):
            df_brasil['people_fully_vaccinated'].iloc[i] = df_brasil['people_fully_vaccinated'].iloc[i-2]
        elif not pd.isnull(df_brasil['people_fully_vaccinated'].iloc[i-3]):
            df_brasil['people_fully_vaccinated'].iloc[i] = df_brasil['people_fully_vaccinated'].iloc[i-3]
        elif not pd.isnull(df_brasil['people_fully_vaccinated'].iloc[i-4]):
            df_brasil['people_fully_vaccinated'].iloc[i] = df_brasil['people_fully_vaccinated'].iloc[i-4]
        else:
            continue
        print(i, 'ok')

    df_brasil.tail(40)

    # Filtrando para dados a partir dos dias em que houveram vacinacao
    df_brasil = df_brasil.query('people_fully_vaccinated > 0').reset_index(drop=True)
    # criando campo para dias percorridos desde a primeira vacinacao
    df_brasil['dia_vacinacao'] = df_brasil.reset_index().reset_index(drop = True)['index']
    

    # criando metrica de porcentagem de vacinados
    df_brasil['per_vacinados'] = df_brasil['people_fully_vaccinated'] / df_brasil['population']

    #df_brasil.corr()

    # ====== Criacao dos vetores ======
    # Y para dias percorridos desde o dia 0,
    # X para porcentagem da populacao vacinada
    X = np.array(df_brasil[['per_vacinados']]).reshape(-1, 1)
    Y = np.array(df_brasil[['dia_vacinacao']]).reshape(-1, 1)

    # ====== Metodo de Regressao ======
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # regressao polinomial grau 3
    pp = PolynomialFeatures(degree=3)
    X_poly = pp.fit_transform(X)

    lm_poly = LinearRegression()

    # ====== Treinando o modelo (criando a funcao) ======
    lm_poly.fit(X_poly, Y)

    # ====== Erro R-score ======
    lm_poly.score(X_poly, Y)

    Y_pred = lm_poly.predict(X_poly)
    
    # ====== Predicoes ======
    # 43% da populacao vacinada
    dia_43_per = lm_poly.predict(pp.fit_transform([[0.43]]))
    # 60% da populacao vacinada
    dia_60_per = lm_poly.predict(pp.fit_transform([[0.6]]))
    # 100% da populacao vacinada
    dia_100_per = lm_poly.predict(pp.fit_transform([[1]]))

    # ====== Plotagem do Grafico ======
    import matplotlib.pyplot as plt 

    plt.plot(X, Y, color='r', label = 'Valores Reais') 
    plt.plot(X, Y_pred, color='g', label = 'Valores Previstos') 
    plt.xlabel("Porcentagem_vacinacao") 
    plt.ylabel("Dias") 
    plt.title("Vacinacao Brasil") 
    plt.legend() 
    plt.savefig('Grafico_projetados_reais.png')
    #plt.show() 
    
    # salvando arquivo
    df_brasil.to_csv('df_trabalho_bigdata.csv', index = False)