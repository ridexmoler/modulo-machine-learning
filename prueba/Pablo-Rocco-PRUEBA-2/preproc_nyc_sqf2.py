#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime


## Helpers


def infer_datatype(df, datatype, drop_none=True):
    """ A partir de un dataset y un tipo de datos entregado, devuelve los nombres de las columnas
        del dataset que tienen el correspondiente tipo de dato.
        
        Argumentos:
           - df: Dataframe de pandas.
           - datatype: String con el tipo de dato que se desea consultar a las columnas del dataframe.
           - drop_none: Filtra las columnas cuyo tipo de dato no esté especificado. default = True.
    """
    tmp_list = [i if df[i].dtype == datatype else None for i in df.columns]
    if drop_none is True:
        tmp_list = list(filter(lambda x: x != None, tmp_list))

    return tmp_list

def return_time_string(var, date_format='%m%d%Y'):
    return var.apply(lambda x: datetime.strptime(str(x), date_format))

def count_freq(df, selected_columns):
    """ Cuenta la cantidad de valores únicos y la frecuencia de dichos valores en las columnas
        entregadas por `selected_columns`.
        
        Argumentos:
            - df: dataframe que contiene las columnas en cuestión.
            - selected_columns: Columnas del dataframe de las que se quiere saber la frecuencia de valores.
    """
    return {i: df[i].unique().shape[0] for i in selected_columns}


def create_suitable_dataframe(df, force_object=[], force_delete=[]):
    """
    Descripción: Crea un dataframe apto para entrenamiento de acuerdo a normas básicas de limpieza de datos faltantes,
        transformación de etiquetas nulas en variables categóricas y crea atributos sinteticos de edad del sospechoso
         y conversión de distancia a sistema metrico.
    Argumentos:
        - df: Un objeto pandas.DataFrame
        - force_object: lista que contiene los nombres de columnas que se forzarán a ser del tipo objet
        - force_delete:  lista que contiene los nombres de columnas que se forzarán a eliminar
    returns: 
    """
    ### Se copia el DataFrame para no afectar el original
    df = df.copy()
    ### Procesamos la eliminación forzada de columnas
    df.drop(columns=force_delete, inplace=True)

    ### Obtener columnas por tipo de dato
    object_data_type = infer_datatype(df, 'object')
    # Se agregan atributos forzados a ser objetos, como objetos
    object_data_type = [objeto for objeto in object_data_type if objeto not in force_object]
    object_data_type = object_data_type + force_object
    
    # Se eliminan atributos forzados a ser objetos
    integer_data_type = infer_datatype(df, 'int')
    integer_data_type = [objeto for objeto in integer_data_type if objeto not in force_object]
    
    # Se eliminan atributos forzados a ser objetos
    float_data_type = infer_datatype(df, 'float')
    float_data_type = [objeto for objeto in float_data_type if objeto not in force_object]
    
    # Quiero recuperar la lista de valores numericos tambien
    suitable_numerical_attributes = list(integer_data_type) + list(float_data_type)
    
    ### Reemplazo de clases faltantes
    ### {N: No, Y: Yes, U: Unknown}
    if 'officrid' in df.columns:
        df['officrid'] = np.where(df['officrid'] == ' ', 'N', 'Y')
    if 'offshld' in df.columns:
        df['offshld'] = np.where(df['offshld'] == ' ', 'N', 'Y')
    if 'sector' in df.columns:
        df['sector'] = np.where(df['sector'] == ' ', 'U', df['sector'])
    if 'trhsloc' in df.columns:
        df['trhsloc'] = np.where(df['trhsloc'] == ' ', 'U', df['trhsloc'])
    if 'beat' in df.columns:
        df['beat'] = np.where(df['beat'] == ' ', 'U', df['beat'])
    if 'offverb' in df.columns:
        df['offverb'] = np.where(df['offverb'] == ' ', 'N', 'Y')
    
    # Se agrega caso ya que no se estaba considerando
    if 'dettypcm' in df.columns:
        df['dettypcm'] = np.where(df['dettypcm'] == ' ', 'CM', df['dettypcm'])
    if 'linecm' in df.columns:
        df['linecm'] = np.where(df['linecm'] == ' ', '1', df['linecm'])
    if 'addrtyp' in df.columns:
        df['addrtyp'] = np.where(df['addrtyp'] == ' ', 'L', df['addrtyp'])

    ### Contar la cantidad de clases en el caso de las var. categóricas y frecuencia de valores para las numéricas
    object_unique_vals = count_freq(df, object_data_type)
    int_unique_vals = count_freq(df, integer_data_type)
    float_unique_vals = count_freq(df, float_data_type)

    ### Selección de atributos categoricos que cumplen con características deseadas
    suitable_categorical_attributes = dict(filter(lambda x: x[1] < 100 and x[1] >= 2, object_unique_vals.items()))
    suitable_categorical_attributes = list(suitable_categorical_attributes.keys())
    
    meters = df['ht_feet'].astype(str) + '.' + df['ht_inch'].astype(str)
    df['meters'] = meters.apply(lambda x: float(x) * 0.3048) # Conversión de distanca a sistema metrico (non retarded)
    df['month'] = return_time_string(df['datestop']).apply(lambda x: x.month) # Agregación a solo meses
    
    # Filtrar solo mayores de 18 años y menores de 100
    df['age_individual'] = np.where(np.logical_and(df['age'] > 18, df['age'] < 100), df['age'], np.nan)
    proc_df = df.dropna()
    preserve_vars = suitable_categorical_attributes + ['age_individual', 'month', 'meters']
    proc_df = proc_df.loc[:, preserve_vars] # Agregar los atributos sintéticos al df
    return proc_df, suitable_categorical_attributes, suitable_numerical_attributes