#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: ancilliary_funcs.py
Author: Pablo Rocco Rosales
Email: pablo[dot]rocco[dot]r[at]gmail[dot]com
Github: https://github.com/ridexmoler
Description: Funciones auxiiares generadas en el desafio 5 del módulo Fundamentos de DS
"""
# Pandas para el manejo de estructuras de datos
import pandas as pd
# Numpy para el manejo de funciones matemáticas
import numpy as np
# Se incorpora librería para generar graficos
import matplotlib.pyplot as plt
# Se incorpora librería para generar graficos
import seaborn as sns
# Metodo para obtener el error cuadrático medio
from sklearn.metrics import mean_squared_error
# Metodo para obtener el error absoluto median
from sklearn.metrics import median_absolute_error
# Metodo para obtener el R-Cuadrado
from sklearn.metrics import r2_score
# Importamos el metodo para obtener los valores de cada atributo del modelo
from pygam.utils import generate_X_grid
# Importamos librerias para metricas de clasificación
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



def report_scores_regressor(ytest, yhat):
    """
    Descripción: Se despliegan métricas ECM, MAE y R^2 de dos vectores
    Entrada:
        ytest, List[float] vector usado para validar un modelo
        yhat, List[float] vector predicho
    Salida:
        void
    """
    m_mse = np.round(mean_squared_error(ytest, yhat), 4)
    m_mae = np.round(median_absolute_error(ytest, yhat), 4)
    m_r2 = np.round(r2_score(ytest, yhat), 2)
    print(f"Error Cuadrático Medio:\n{m_mse}\n")
    print(f"Error Medio Absoluto:\n{m_mae}\n")
    print(f"R cuadrado:\n{m_r2}");


def binarize_column(df, columns):
    """
    Descripción: Se recibe un DataFrame [df] y se binarizan las columnas listadas en [columns]
    Entrada: 
        df, DataFrame que contiene los datos, debe contener las columnas [columns]
        columns, List que almacena el nombre de las columnas que serán binarizadas
    retorno:
        df, Dataframe recibido por parámetro y que adiciona las nuevas columnas binarizadas 
    """
    for col in columns:
        order_column = df[col].value_counts(ascending=True)
        if len(order_column)>1:
            for index in range(len(order_column)-1):
                sufix = order_column.index[index].lower()
                df[col.lower()+'_'+sufix] = np.where(df[col]==order_column.index[index], 1, np.where(df[col].isnull(), np.nan, 0))
    return df


def replace_values_column(df, column, replacement):
    """
    Descripción: Se recibe un DataFrame [df] y se reemplazan los valores de la columna [column]
    respecto a los valores contenidos en el diccionario [replacement]
    Entrada: 
        df, DataFrame que contiene los datos, debe contener la columna [column]
        column, nombre de la columna que será procesada
        replacement, estructura dict que mapea la clave como el valor reemplazado y
        valor como el valor a reemplazar
    retorno:
        df[column], Serie que almacena la columna reemplazada
    """
    for clave, valor in replacement.items():
        df[column] = df[column].replace(valor, clave)
    return df[column]
    
def stats_no_use_describe(datos):
    """
    Descripción: Por cada variable existente en el objeto [datos], se calculan las medidas descriptivas 
    para los casos contínuos y para cada variable discreta se calcula la frecuencia.
    imprime en pantalla los estadisticos de las variables seleccionadas 'gle_cgdpc' , 'undp_hdi' , 'imf_pop' 
    Entrada: 
        datos, Un objeto DataFrame que contiene atributos de un set de observaciones
    retorno: void 
    *NO usa el metodo describe()
    """
    
    # Se almacenan las columnas a estudiar
    lista_a_reportar = ['gle_cgdpc' , 'undp_hdi' , 'imf_pop']
    
    # Se recorre por columna para realizar el computo de los estadisticos descriptivos
    for columna, serie in datos.items():
        if serie.dtype == 'object':
            frecuencia = serie.dropna().value_counts()
            if columna in lista_a_reportar:
                print(f"Columna {columna}: Frecuencia = {serie.dropna().value_counts()}")
        elif serie.dtype in ['float', 'int']:
            mu = serie.dropna().mean()
            var = serie.dropna().var()
            dstd = serie.dropna().std()
            if columna in lista_a_reportar:
                print(f"Columna {columna}: media = {mu} , var = {var} y dstd = {dstd}")
        else:
            print(f"Tipo de datos [{serie.dtype}] no implementado")


def stats_use_describe(datos):
    """
    Descripción: Por cada variable existente en el objeto [datos], se calculan las medidas descriptivas 
    para los casos contínuos y para cada variable discreta se calcula la frecuencia.
    imprime en pantalla los estadisticos de las variables seleccionadas 'gle_cgdpc' , 'undp_hdi' , 'imf_pop'
    Entrada: 
        datos, Un objeto DataFrame que contiene atributos de un set de observaciones
    retorno: void
    *usa el metodo describe()
    """
    
    # Se almacenan las columnas a estudiar
    lista_a_reportar = ['gle_cgdpc' , 'undp_hdi' , 'imf_pop']
    
    # Se recorre por columna para realizar el computo de los estadisticos descriptivos
    for columna, serie in datos.items():
        if columna in lista_a_reportar:
            print(f"Columna {columna}: medidas\n{serie.describe()}")


def list_na(dataframe, var, print_list = False):
    """
    Descripción: Se analizan los datos perdidos de la variable [var] de un [dataframe],
    se entrega la cantidad y porcentaje de datos perdidos y cuando se requiere [print_list]
    se entrega la lista de datos perdidos
    
    Entrada:
        dataframe, Un objeto DataFrame que contiene atributos de un set de observaciones
        var, Un string que almacena el nombre de la columna a inspeccionar (variable a observar)
        print_list, Un boolean que determina si se retorna la lista de los casos NaN, por defecto es False
    
    retorno (estadisticos):
        cantidad, float con la cantidad de datos perdidos
        porcentaje, float con el porcentaje que representan los datos perdidos de la columna [var]
        lista, listado de los casos perdidos
    """
    
    # cantidad de datos na en la variable a analizar
    cantidad = dataframe[var].isna().sum()
    # porcentaje de datos na en la variable a analizar
    porcentaje = (dataframe[var].isna().mean())*100
    if print_list:
        # lista de elementos con na como valor. La idea de no entregar una columna específica
        # como el cname, es para generalizar a cualquier set de datos.
        lista = dataframe[var].isna()
        return cantidad, porcentaje, lista
    else:
        return cantidad, porcentaje


def gfx_dotplot(dataframe, plot_var, plot_by, global_stat=False, statistic="mean"):
    """
    Descripción: Genera un gráfico de puntos de las medias/mediana de la variable plot_var,
    agrupando por la variable plot_by.
    En caso de que se solicite [global_stat] se genera una linea vertical
    con la media/mediana de la observación
    Entrada:
        dataframe, Un objeto DataFrame que contiene atributos de un set de observaciones
        plot_var, Un string que almacena el nombre de la columna a graficar (variable a observar)
        plot_by, Un string que almacena el nombre de la columna que se debe agrupar.
        global_stat, Un booleano. Si es verdadero, debe generar una recta vertical indicando
        la media de variable en la base de datos.
        statistic, string que debe presentar dos opciones. mean para la media y median para la mediana. 
        Por defecto debe ser mean.
    retorno:
        void
    """
    
    # Si se solicita la media
    if statistic == "mean":
        groupby_statistic = dataframe.groupby(plot_by)[plot_var].mean()
        if global_stat:
            global_value = dataframe[plot_var].mean()
            label_stat = f"Mean: {round(global_value, 2)}"
    # Si se solicita la mediana
    elif statistic == "median":
        groupby_statistic = dataframe.groupby(plot_by)[plot_var].median()
        if global_stat:
            global_value = dataframe[plot_var].median()
            label_stat = f"Median: {round(global_value, 4)}"
    # En cualquier otro caso
    else:
        print(f"Estadistico [{statistic}] no implementado")
        return
    # Se genera el gráfico de puntos, con los valores de las medias de la variable agrupada
    plt.plot(groupby_statistic.values, groupby_statistic.index, 'o')
    plt.title(statistic + ': ' + plot_var + ' (group by ' + plot_by + ')')
    
    # Si se solicita se genera la linea vertical en la posicion de la media de la base completa
    if global_stat:
        plt.axvline(global_value, lw=1, color='tomato', linestyle='--', label=label_stat)
        plt.legend(loc='upper right')


def gfx_hist(dataframe, var, sample_mean=False, true_mean=False):
    """
    Descripción: grafica un histograma y señala las medias con una linea vertical
    En caso de que se solicite [sample_mean]. Y si se requiere se genera otra
    linea vertical con la media de la observación completa [true_mean]
    Entrada:
        dataframe, Un objeto DataFrame que contiene atributos de un set de observaciones
        var, Un string que almacena el nombre de la columna a graficar (variable a observar)
        sample_mean, Un booleano. Si es verdadero, debe generar una recta vertical indicando 
        la media de la variable en la selección muestral. Por defecto debe ser False .
        true_mean, Un booleano. Si es verdadero, debe generar una recta vertical indicando
        la media de variable en la base de datos completa.
    retorno:
        void
    """
    
    # Se selecciona la serie de la variable a observar y se limpian los NA
    muestra_dropna = dataframe[var].dropna()
    # Se genera el histograma de la variable [var]
    plt.hist(muestra_dropna, color = 'lightgrey')
    # Se agrega un titulo al gráfico
    plt.title('Histograma: ' + var)
    if sample_mean:
        # Se obtiene la media de la muestra
        media_muestra = muestra_dropna.mean()
        # Se crea la etiqueta para mostrar en el gráfico
        label_media_muestra = f"Media submuestra: {round(media_muestra,2)}"
        # Si se solicita se genera la linea vertical en la posicion de la media de la muestra
        plt.axvline(media_muestra, lw=1, color='red', linestyle='--', label=label_media_muestra)
    if true_mean:
        # Se limpian los NaN de la variable de la muestra completa
        dropna = df[var].dropna()
        # Se obtiene la media de la muestra completa
        media_todos = dropna.mean()
        # Se crea la etiqueta para mostrar en el gráfico
        label_media_todos = f"Media Data Completa: {round(media_todos,2)}"
        # Si se solicita se genera la linea vertical en la posicion de la media de la base completa
        plt.axvline(media_todos, lw=1, color='blue', linestyle='-.', label=label_media_todos)
    if sample_mean or true_mean:
        plt.legend(loc='upper right')


def display_boxplot(df, variables):
    """
    Descripción: Se despliega un gráfico de caja para todas las [variables] de [df] (DataFrame)
    Entrada:
        df, Dataframe que contiene nuestras variables a observar
        variables, List[String] lista de variables a graficar
    Salida:
        void
    Version: 1
    Date: 07/27/2019
    """
    n_columns = len(variables)
    max_columns = 4
    
    df_gx = df.loc[:,variables]
    
    if n_columns == 1:
        plt.figure(figsize=(3, 5))
        sns.boxplot(y=variables[0], data=df_gx, width=.2, color='#56B4E9', linewidth=1, fliersize=3)
    elif n_columns <=max_columns:
        ancho = np.min([n_columns * 3, 15])
        fig, ax = plt.subplots(1, n_columns, figsize=(ancho, 5))
        for i, var in enumerate(df_gx):
            sns.boxplot(y=var, data=df_gx, ax=ax[i], width=.2, color='#56B4E9', linewidth=1, fliersize=3)
    else:
        rows = int(n_columns / max_columns)
        if n_columns % max_columns != 0:
            rows = rows + 1
        fig, ax = plt.subplots(rows, max_columns, figsize=(15, 10))
        for i, var in enumerate(df_gx):
            row = (int(i/max_columns))
            col = (int(i%max_columns))
            sns.boxplot(y=var, data=df_gx, ax=ax[row, col], width=.2, color='#56B4E9', linewidth=1, fliersize=3)
    plt.tight_layout()


def display_outliers(df):
    """
    Descripción: Se obtienen los datos atípicos de todas las variables incluidad en el [df]
    Entrada:
        df, Dataframe que contiene nuestras variables a observar
    Salida:
        outliers, DataFrame que contiene el listado de variables, la cantidad de datos por variable, 
        la cantidad de datos atípicos y el porcentaje
    Version: 1
    Date: 07/27/2019
    """
    len_df = len(df)
    name = []
    val_n = []
    val_p = []
    val_total = []
    # Recorro la matriz que contiene informaión descriptiva de nuestra data
    for var, serie in df.describe().iteritems():
        # Rango entre el cuartil 1 y 3
        ric = serie['75%'] - serie['25%']
        # Limite inferior para determinar los datos atípicos
        inf = serie['25%'] - ric*1.5
        # Límite superior para determinar los datos atípicos
        sup = serie['75%'] + ric*1.5
        # con len() obtengo la cantidad de datos que escapan de los limites inf y sup
        n_outliers = len(df[(df[var] < inf) | (df[var] > sup)])
        # obtengo el % de datos atípicos de la variable
        perc_outliers = np.round(100*(n_outliers/len_df), 2)
        # Reporto el % de datos atípicos de la variable
        name.append(var)
        val_n.append(n_outliers)
        val_p.append(perc_outliers)
        val_total.append(int(serie['count']))
    # DataFrame que contiene el listado de variables, la cantidad de datos por variable, 
    # la cantidad de datos atípicos y el porcentaje
    outliers = pd.DataFrame({"Variable": name, "count": val_total, "outliers": val_n, "outliers %": val_p}) 
    return outliers


def report_regularization(model, X_test, ytest, show_mse_cv = False):
    """
    Descripción: Se genera una estimación [yhat] con el [modelo] y nuestros datos de prueba [X_test]
    luego se utiliza los valores reales de nuestro vector objetivo [ytest] para desplegar métricas ECM, MAE y R^2
    Entrada:
        model, modelo entrenado previamente
        X_test, Matriz con los atributos de prueba  
        ytest, List[float] vector usado para validar el [modelo]
    Salida:
        void
    """
    print(f"Valor del parámetro de regularización: {model.alpha_}\n")
    print(f"Coeficientes finales: \n{pd.Series(model.coef_, X_test.columns)}\n")
    yhat =  model.predict(X_test)
    if show_mse_cv is True:
        # Reportamos el error (metrica) en el proceso de Regularizacion con CV
        print(f"Valores de la métrica de desempeño (mse) de CV:")
        cv_values = pd.DataFrame(model.mse_path_)
        cv_values.columns = ['cv_' + str(x) for x in range(1, model.cv+1)]
        cv_values.index = ['alpha_' + str(x) for x in model.alphas]
        display(cv_values)
    report_scores_lineal(ytest, yhat)


def display_partial_dependence(model, X_train, marker = '|', color = 'dodgerblue'):
    """
    Descripción: Se despliega una grilla de con todos los gráficos de dependencia 
    parcial de los atributos
    Entrada:
        model, modelo entrenado previamente
        X_train, DataFrame con los atributos de entrenamiento
    Salida:
        void
    Version: 1
    Date: 07/27/2019
    """
    x_grid = generate_X_grid(model)
    # Listamos los atributos
    variables = X_train.columns
    # generamos el dimensionado del grid en base a la cantidad de atributos
    n_columns = len(variables)
    # Fijamos un máximo de 4 columnas de gráficos
    max_columns = 4
    # Calculamos en base a la cantodidad maxima de columnas, las filas
    rows = np.ceil(n_columns /max_columns)
    # Fijamos el tamaño del paño de los gráficos
    plt.figure(figsize=(15, 10))
    # Generamos un gráfico de dependencias parciales para cada variable
    for i, name in enumerate(variables):
        plt.subplot(rows, max_columns, i + 1)
        # extraemos la dependencia parcial y sus intervalos de confianza al 95%
        partial_dep, confidence_intervals = model.partial_dependence(x_grid, feature= i + 1, width=0.95)
        # Generamos la linea que describe la curva
        plt.plot(x_grid[:, i], partial_dep, color=color)
        # Generamos una visualización de los intervalos de confianza
        plt.fill_between(x_grid[:, i], 
                         confidence_intervals[0][:, 0], 
                         confidence_intervals[0][:, 1],
                         color = color,
                         alpha = .25)
        x_vect = X_train[name]
        y_vect = [plt.ylim()[0]] * len(X_train[name])
        plt.scatter(X_train[name], y_vect, marker=marker, color = 'orange', alpha = .5, s=500)
        # agregamos el nombre del atributo
        plt.title(name)
        plt.tight_layout()


def plot_confusion_matrix(y_test, y_hat, classes_labels):
    """
    Descripción: Se despliega graficamente la matriz de confusión
    Entrada:
        y_test, Vector con las clases reales
        y_hat, Vector con las clases predichas
        classes_labels, Etiquetas de las clases
    Salida:
        void
    Version: 1
    Date: 08/03/2019
    """
    tmp_confused = confusion_matrix(y_test, y_hat)
    sns.heatmap(tmp_confused, annot=True, cbar=False, xticklabels=classes_labels,
                yticklabels=classes_labels, cmap='Greens', fmt=".1f")
    plt.xlabel('Classes on testing data')
    plt.ylabel('Predicted classes on training')
    plt.grid(False)


def plot_class_report(y_test, y_hat, classes_labels):
    """
    Descripción: Se despliega gráficamente el reporte de classification_report
    Entrada:
        y_test, Vector con las clases reales
        y_hat, Vector con las clases predichas
        classes_labels, Etiquetas de las clases
    Salida:
        void
    Version: 1
    Date: 08/03/2019
    """
    tmp_report = classification_report(y_test, y_hat, output_dict=True)
    targets = list(classes_labels)
    targets.append('average')
    tmp_report = pd.DataFrame(tmp_report)\
                    .drop(columns=['weighted avg', 'macro avg'])
    tmp_report.columns = targets
    tmp_report = tmp_report.drop(labels='support')
    tmp_report = tmp_report.drop(columns='average')
    tmp_report = tmp_report.T

    for index, (colname, serie) in enumerate(tmp_report.iteritems()):
        plt.subplot(3, 1, index + 1)
        serie.plot(kind = 'barh')
        plt.title(f"Métrica: {colname}")
        plt.tight_layout()


def plot_importance(fit_model, feat_names, top = 10):
    """
    Descripción: Grafica el nivel de importancia de los atributos en un [fit_model], despliega
    un listado limitado por el [top] o la cantidad máxima de atributos
    Entrada:
        fit_model: Es un objeto de tipo modelo de sklearn, del cual extraemos el atributo feature_importances_
        feat_names: Lista de String  de los Nombres de los atributos
        top: límite de despliegue de atributos
    Salida:
        void
    Version: 1
    Date: 08/22/2019

    """
    # Se obtienen un arreglo con los valores de immportancia de cada atributo
    tmp_importance = fit_model.feature_importances_
    # Se ordenan los indices de la lista en relación al orden descendente de
    # los valores de arreglo (importancia de los atributos).
    sort_index_importance = np.argsort(tmp_importance)[::-1]
    # Se verifica que el top solicitado no supere la cantidad de atributos, en
    # caso de que ocurra se fija en la cantidad de atributos
    if tmp_importance.shape[0] < top:
        top = tmp_importance.shape[0]
    # Se seleccionan los indices de los n-top atributos
    sort_index_importance = sort_index_importance[0:top]
    # Se seleccionan los nombres de los atributos con los indices antes guardados
    names = [feat_names[i] for i in sort_index_importance]
    # Titulo del gráfico
    plt.title(f"Feature importance - Top {top}")
    # Se genera un gráfico de barras horizontales
    plt.barh(range(len(names)), tmp_importance[sort_index_importance].tolist())
    # Se etiquetan las barras con los nombres de los atributos
    plt.yticks(range(len(names)), names, rotation=0)

def plot_roc(model, X_test, y_test, model_label=None):
    """
    Descripción: Grafica el area bajo la curva roc
    Entrada:
        model: Es un objeto de tipo modelo de sklearn
        X_test: Matriz de atributos usados en la prediccion
        y_test: Vector objetivo con valores reales
        model_label: String nombre de la etiqueta de la serie
    Salida:
        void
    Version: 1
    Date: 08/30/2019
    """
    tmp_y_pred = model.predict_proba(X_test)[:, 1]
    false_positive_rates, true_positive_rates, _ = roc_curve(y_test, tmp_y_pred)
    store_auc = auc(false_positive_rates, true_positive_rates)
    if model_label is not None:
        tmp_label = "{}: {}".format(model_label, round(store_auc,3))
    else:
        tmp_label = None
    plt.plot(false_positive_rates, true_positive_rates, label=tmp_label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')