import pandas as pd
import numpy as np
from datetime import datetime

from sklearn import linear_model, preprocessing, metrics, model_selection


import matplotlib.pyplot as plt
import seaborn as sns

# Funciones varias de uso general

def reduce_mem_usage(df, obj2category=True, verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.     

        parameters:
            df: pd.DataFrame to process
            obj2category: (boolean, default=True) convert object columns to category
            verbose: (boolean, default=True) inform the memory usage before, after and optimiztion percentage  
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if obj2category:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def describe(df_or_serie):
    """
    Parameter: df_or_serie, a pd.DataFrame or pd.Series to describe
    Returns a DataFrame with a row for every column of the parameter df and several columns with descriptive stats
    columns in the DataFrame returned:
        dtype : Data type of the column
        count : # of non null values in column
        nulls : # of null values
        null_porc : % of null values
        unique : # of unique values in column
        mean : mean (only numeric)
        std : std deviation (only numeric)
        min : minimum value (only numeric)
        5%, 25%, 50%, 75%, 95% : some useful percentiles (only numeric)
        max : maximum value (only numeric)
    """
    df = df_or_serie
    if type(df) is not pd.DataFrame:
        df = pd.DataFrame(df)
    dtypes = df.dtypes.rename('dtype')
    count = df.count().rename('count')
    nulos = df.isnull().sum().rename('nulls')
    porc_nulos = ( np.round(100 * df.isnull().sum() / df.shape[0]) ).round(2).rename('null_porc')
    describes = df.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).round(3).transpose()
    describes.drop(columns='count', inplace=True)
    if 'unique' in describes.columns:
        describes.drop(columns='unique', inplace=True)
    uniques = df.nunique().rename('unique') # calculate 'unique' even for numeric columns
    return pd.concat([dtypes, count, nulos, porc_nulos, uniques, describes], axis='columns')


def valCounts(dfOrSerie, **kwargs):
    """
    Returns a DataFrame with all unique values of the pd.Series or pd.DataFrame, with 4 columns:
        - count: number or occurrences
        - perc: percentaje of the occurrences of that value calculated over the TOTAL lenght of the Series/DataFrame
        - cumsum: cumulative sum of the count column
        - cum_perc: cumulative sum of the perc column
    It adds a row with the null values count, perc, etc. at the end
    **kwargs is any parameter to be pased to pd.Series.value_counts() or pd.DataFrame.value_counts() function
    """
    count     = dfOrSerie.value_counts(**kwargs)
    count     = count.append( pd.Series(len(dfOrSerie)-count.sum(), index=['< Null values >']) ).rename('count').astype(int)
    cumsum    = count.cumsum().rename('cumsum').astype(int)
    perc      = np.round( 100 * count / len(dfOrSerie), 1 ).rename('perc').astype(float)
    cum_perc  = np.round( 100 * cumsum / len(dfOrSerie), 1 ).rename('cum_perc').astype(float)
    return pd.concat( [ count, perc, cumsum, cum_perc ], axis='columns' )


def concurrentMasks(mask_list, symbols=(True,False), ascending=True):
    """
    Makes an ordered table of how many records share each one of the mask values in the list
    Parameters:
        mask_list: A list of masks. They must have all the same length and all be based on the same DataFrame
        symbols: A list-like with exactly two elements, they're shown in the table when the mask value is 
                 respectively True and False
        ascending: If True (default) sorts the result in ascending order, otherwise in descending order
    Returns a DataFrame containing the report
    """
    ret = pd.concat(mask_list, axis='columns').value_counts().rename('Qty')
    ret = ret.reset_index()
    ret.replace(to_replace=True, value=symbols[0], inplace=True)
    ret.replace(to_replace=False, value=symbols[1], inplace=True)
    ret.sort_values(by=list(ret.columns), ascending=ascending, inplace=True)
    return ret


def seriesTranspose( serie ):
    """
    Return a DataFrame with the transposed series received as a parameter
    Note that pd.Series.transpose() returns itself, because a Series is a one-dimensional ndarray, anyway sometimes it's
    useful to have the Series as a one row DataFrame with the data in columns and the index data as the column names.
    """
    return pd.DataFrame(data=[serie.values], columns=serie.index)


def outlierLims(serie, level=1.5):
    """
    Devuelve una tupla con los limites inferior y superior de la zona de no-outliers correspondientes a los datos de la serie
    serie: pd.Series conteniendo los datos
    level: nivel de los outliers, cantidad de rangos intercuartiles por debajo de q1 o 
           por encima de q3 que tiene que estar un dato para ser considerado outlier (default 1.5, 
           valor usual para outliers extremos 3.0)
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    return ( q1 - level*(q3-q1), q3 + level*(q3-q1) )


def outlierMask(serie, direc='b', level=1.5, min=None, max=None):
    """
    Devuelve una mascara booleana correspondiente a los datos de la serie que son outliers
    serie: pd.Series conteniendo los datos
    direc: 'l' para los outliers por defecto (lower), 'h' para los outliers por exceso (higher), 'b' para ambos (both)
    level: nivel de los outliers, cantidad de rangos intercuartiles por debajo de q1 o 
           por encima de q3 que tiene que estar un dato para ser considerado outlier (default 1.5, 
           valor usual para outliers extremos 3.0)
    min: si es nulo (default) el limite inferior es q1 - level * riq, si se especifica, ignora eso y el limite inferior es min
    min: si es nulo (default) el limite superior es q3 + level * riq, si se especifica, ignora eso y el limite superior es max
    """
    low, high = outlierLims(serie,level)
    low  = low  if min is None else min
    high = high if max is None else max
    if direc == 'l':
        return ( serie < low )
    elif direc == 'h':
        return ( serie > high )
    else:
        return ( serie < low ) | ( serie > high )


from sklearn.base import BaseEstimator, TransformerMixin

class ColSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, verbose=0):
        """
        columns: lista de columnas a seleccionar
        verbose: nivel de verbose (0: nada, 1: X.shape y len(columns), 2: lo mismo mas el nombre de las columnas)
        (TODO: En un futuro estaría bueno agregar acá algun parametro para seleccionar por dtype o alguna otra condicion)
        """
        self.columns = columns
        self.verbose = verbose
    
    def transform(self, X, y=None):
        if self.verbose > 0:
            print('ColSelector.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape), ' - len(columns)', len(self.columns))
        if self.verbose > 1:
            print('columns:', self.columns)
        if type(X) is pd.DataFrame:
            return X[self.columns]
        else: 
            return pd.DataFrame(X)[self.columns] # si lo llaman con una serie la transformo primero en DataFrame
    
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('ColSelector.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape), ' - len(columns)', len(self.columns))
        if self.verbose > 1:
            print('columns:', self.columns)
        return self

class MakeBins(BaseEstimator, TransformerMixin):
    
    def __init__( self, dicts, verbose=0 ):
        """
        Crea bins para todas las columnas en el diccionario pasado por parametro
        
        Parametros:
            dicts: diccionario con una entrada por cada columna a transformar. Para cada columna se especifica
                   un diccionario con uno o mas de las siguientes entradas:
                      'bins': <bins a pasar como parametro a pd.cut> (obligatorio)
                      'kwargs': <diccionario con los argumentos opcionales para la funcion pd.cut>
                      'astypeStr': True o False segun si se quiere convertir la columna a str luego de hacer los bins
                      'outCategory': Valor a colocar en caso de que un dato no encaje en ningun bin
                ej: {
                        'columna_1': 
                            { 'bins': [0,30,40,90,120],
                              'astypeStr': True,
                              'outCategory': 'fuera de rango'
                            },
                        'columna_2': 
                            { 'bins': [0,50,100,200],
                              'kwargs': {'labels': ['small','medium','large']},
                              'outCategory': 'large'
                            }
                    }
            Los valores no encontrados en el diccionario correspondiente a cada columna son reemplazados por np.NaN
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.dicts = dicts
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('MakeBins.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        return self
    
    def transform(self, X, y=None):

        if self.verbose > 0:
            print('MakeBins.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        if type(X) is not pd.DataFrame: # si lo llamaron con una serie, la convierto a DataFrame para hacer el proceso
            data = pd.DataFrame(X)
        else:
            data = X
        
        if self.verbose > 1:
            print('Columns: ', end='')
        cols = []
        for col in data.columns.values:
            
            if col not in self.dicts:    # Si no es una de las columnas a procesar, la dejo intacta
                cols.append( data[col] )
                continue

            if self.verbose > 1:
                print(col, end=' ')

            astypeStr   = False if 'astypeStr'   not in self.dicts[col] else self.dicts[col]['astypeStr']
            outCategory = None  if 'outCategory' not in self.dicts[col] else self.dicts[col]['outCategory']
            kwargs      = {}    if 'kwargs'      not in self.dicts[col] else self.dicts[col]['kwargs']

            c = pd.cut( data[col], self.dicts[col]['bins'], **kwargs )
            # si outCategory no pertenece a las categorias generadas hay que pasar todo a str antes de poder asignarla
            if astypeStr: 
                c = c.astype(str) # los np.NaN quedaran como 'nan'
                if outCategory != None:
                    c = c.replace('nan',outCategory)
            else:
                if outCategory != None:
                    c = c.fillna(outCategory)
                
            cols.append( c )

        if self.verbose > 1:
            print()
            
        if type(X) is not pd.DataFrame: # si era una serie devuelvo una serie
            return cols[0]
        else: # si no, devuelvo un DataFrame
            return pd.concat( cols, axis = 'columns' )

class MapDict(BaseEstimator, TransformerMixin):
    
    def __init__( self, dicts, verbose=0 ):
        """
        Cambia los valores de cada columna según los valores de un diccionario
        
        Parametros:
            diccionario de diccionarios. La clave del diccionario es el nombre de la columna a transformar. Para cada columna se especifica un diccionario donde cada valor
            posible que tome esa columna se corresponde con uno a reemplazar
                ej: {'columna_1':
                         {'valor A': 1,
                          'valor B': 2,
                          'valor C': 3},
                     'columna_2':
                         {'valor D': 'reemplazo X',
                          'valor E': 'reemplazo Y'}
                    }
            Los valores no encontrados en el diccionario correspondiente a cada columna son reemplazados por np.NaN
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.dicts = dicts
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('MapDict.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        return self
    
    def transform(self, X, y=None):

        if self.verbose > 0:
            print('MapDict.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        if type(X) is not pd.DataFrame: # si lo llamaron con una serie, la convierto a DataFrame para hacer el proceso
            data = pd.DataFrame(X)
        else:
            data = X

        if self.verbose > 1:
            print('Columns: ', end='')
        cols = []
        for col in data.columns.values:

            if col not in self.dicts:    # Si no es una de las columnas a procesar, la dejo intacta
                cols.append( data[col] )
                continue

            if self.verbose > 1:
                print(col, end=' ')
            
            cols.append( data[col].map( self.dicts[col] ).fillna(data[col]) )

        if self.verbose > 1:
            print()
        
        if type(X) is not pd.DataFrame: # si era una serie devuelvo una serie
            return cols[0]
        else: # si no, devuelvo un DataFrame
            return pd.concat( cols, axis = 'columns' )

class FillNulls(BaseEstimator, TransformerMixin):
    
    def __init__( self, nulls, verbose=0 ):
        """
        Hace un fillna() con un valor especifico para cada columna
        
        Parametros:
            nulls: diccionario cuya clave es el nombre de la columna y el valor es el dato colocar en el fillna()
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.nulls = nulls
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('FillNulls.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        return self
    
    def transform(self, X, y=None):

        if self.verbose > 0:
            print('FillNulls.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        if type(X) is not pd.DataFrame: # si lo llamaron con una serie, la convierto a DataFrame para hacer el proceso
            data = pd.DataFrame(X)
        else:
            data = X
        
        if self.verbose > 1:
            print('Columns: ', end='')
        cols = []
        for col in data.columns.values:
            
            if col not in self.nulls:    # Si no es una de las columnas a procesar, la dejo intacta
                cols.append( data[col] )
                continue

            if self.verbose > 1:
                print(col, end=' ')
            
            cols.append( data[col].fillna( self.nulls[col] ) )

        if self.verbose > 1:
            print()
        
        if type(X) is not pd.DataFrame: # si era una serie devuelvo una serie
            return cols[0]
        else: # si no, devuelvo un DataFrame
            return pd.concat( cols, axis = 'columns' )
    
class DFTransform(BaseEstimator, TransformerMixin):
    
    def __init__( self, transformer, verbose=0 ):
        """
        Transforma un DataFrame con un transformer dado, conservando los nombres de las columnas y los indices
        Esto lo hago para evitar perder esa info al usar el StandardScaler u otros que te devuelven un ndarray 'pelado'
        
        Parametros:
            transformer: el transformer que se usara
            verbose: nivel de verbose (0: nada, 1: X.shape)
        """
        self.transformer = transformer
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('DFTransform.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        self.transformer.fit(X)
        return self
    
    def transform(self, X, y=None):
        if self.verbose > 0:
            print('DFTransform.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        return pd.DataFrame( self.transformer.transform(X), index=X.index, columns=X.columns )

class DFUnion(BaseEstimator, TransformerMixin):
    
    def __init__( self, pipelines, verbose=0 ):
        """
        Une las columnas provenientes de distintos pipelines, similar a sklearn.pipeline.FeatureUnion solo que:
            - No tiene las opciones de procesamiento paralelo, pero...
            - Conservando los nombres de las columnas y los indices
        Si estoy usando el pipeline para procesar el dataset y voy a correr algun tipo de feature importance o necesito poder explicar
        el modelo entrenado, quiero conservar los nombres de las columnas, en caso contrario usar el FeatureUnion estandar
        
        Parametros:
            pipelines: lista de tuplas ('nombre', pipe) a la manera de las que se usan para definir un pipeline. Se ejecutaran todos
                       los pipelines de la lista, c/u debe devolver un DataFrame, y luego se devolvera la union de todos en uno unico
            verbose: nivel de verbose (0: nada, 1: X.shape (in) + df.shape (out))
        """
        self.pipelines = pipelines
        self.verbose = verbose
        
    def fit(self, X, y=None):

        if self.verbose > 0:
            print('DFUnion.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        for p in self.pipelines:
            p[1].fit(X,y)
        return self
    
    def transform(self, X, y=None):

        if self.verbose > 0:
            print('DFUnion.transform (in) - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        
        union = []
        for p in self.pipelines:
            df = p[1].transform(X)
            if type(df) is not pd.DataFrame: # si el pipeline devolvio un ndarray, lo convierto a DataFrame
                df = pd.DataFrame(df)
            union.append( df )

        ret = pd.concat( union, axis='columns' )

        if self.verbose > 0:
            print('DFUnion.transform (out)', ret.shape)

        return ret

class Verbose(BaseEstimator, TransformerMixin):
    
    def __init__( self, text='', nivel=0 ):
        """
        Imprime data de X e y para monitorear el avance del pipeline
        
        verbose: nivel de verbose (0: nada, 1: X.shape + y.shape (si fue pasado), 2: lo mismo mas X.columns)
        """
        self.nivel = nivel
        self.text = text
        
    def fit(self, X, y=None):
        if self.nivel > 0:
            print( f'{self.text} fit X.shape {X.shape}, y.shape { None if y is None else y.shape }' )
        if self.nivel > 1:
            print( f'{self.text} fit columns=', X.columns )
        return self
    
    def transform(self, X, y=None):
        if self.nivel > 0:
            print( f'{self.text} transform X.shape {X.shape}, y.shape { None if y is None else y.shape }' )
        if self.nivel > 1:
            print( f'{self.text} transform columns=', X.columns )
        if y is None:
            return X
        else:
            return X, y

class OutOfRange2NaN(BaseEstimator, TransformerMixin):
    
    def __init__( self, ranges, verbose=0 ):
        """
        Convierte en  np.NaN todos los valores que esten fuera del rango especificado
        
        Parametros:
            ranges: diccionario cuya clave es el nombre de la columna y el valor una tupla con el rango
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.ranges = ranges
        self.verbose = verbose
        
    def fit(self, X, y=None):
        if self.verbose > 0:
            print('OutOfRange2NaN.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        return self
    
    def out2NaN( self, x, r ):
        if x is np.NaN:
            return np.NaN
        if x < r[0] or x > r[1]:
            return np.NaN
        else:
            return x
    
    def transform(self, X, y=None):
        if self.verbose > 0:
            print('OutOfRange2NaN.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        if type(X) is not pd.DataFrame: # si lo llamaron con una serie, la convierto a DataFrame para hacer el proceso
            data = pd.DataFrame(X)
        else:
            data = X
        
        if self.verbose > 1:
            print('Columns: ', end='')
        cols = []
        for col in data.columns.values:
            
            if col not in self.ranges:    # Si no es una de las columnas a procesar, la dejo intacta
                cols.append( data[col] )
                continue
            
            if self.verbose > 1:
                print(col, end=' ')

            cols.append( data[col].apply( lambda x: self.out2NaN(x,self.ranges[col])  ) )
        
        if self.verbose > 1:
            print()

        if type(X) is not pd.DataFrame: # si era una serie devuelvo una serie
            return cols[0]
        else: # si no, devuelvo un DataFrame
            return pd.concat( cols, axis = 'columns' )
    
def text2name(txt):
    return ''.join(char if char.isalnum() else '_' for char in txt)

class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 minQty = 0, 
                 minPerc = 0, 
                 otherPerc = 0, 
                 maxCategs = None,
                 column_list = None, 
                 pref_list = None, 
                 verbose = 0
                ):
        """
        Genera un encoder para cada columna del DataFrame o Serie. Permite limitar las categorias y englobar 
        las que no queden explicitas para que queden como "otros"

        En caso de haber filtrado categorias (por no entrar en los minimos especificados por parametro), ceros en todas las columnas indica 'otros'
        En caso de no haber filtrado ninguna categoria, se dropea la columna mayoritaria
        Para cada columna a procesar guarda el nombre de la categoria dropeada en el atributo drop_class

        Parametros:
            minQty: minima cantidad de ocurrencias para que una categoria se transforme en categoria explicita
            minPerc: minimo porcentaje del total para que una categoria individual se tranforme en categoria explicita
            otherPerc: porcentaje acumulado de registros que no se codificaran y pasaran a formar parte de "otros"
            maxCategs: maxima cantidad de categorias a generar (las otras quedan como "otros")
            column_list:  lista de columnas a transformar. None (default) para transformar todas las de tipo 'object' o 'category'
            pref_list: lista de prefijos a utilizar en cada columna. None (default) para usar el nombre de la columna de texto como prefijo. 
                       Las columnas dummies tendrán como nombre el prefijo + el texto de la opcion (limpio de caracteres especiales)
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.minQty = minQty
        self.minPerc = minPerc
        self.otherPerc = otherPerc
        self.maxCategs = maxCategs
        self.column_list = column_list
        self.pref_list = pref_list
        self.verbose = verbose
    
    def fit( self, X, y=None ):
        
        if self.verbose > 0:
            print('LabelEncoder.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))

        data = pd.DataFrame(X) if type(X) is pd.Series else X # Si fue llamada con una serie, la transformo en DataFrame

        if self.column_list is None: # Si no se especifico una lista de columnas, tomo todas las columnas de tipo object o category
            self.column_list = list( data.dtypes [ data.dtypes.apply( lambda x: str(x) in ['object','category'] )  ].index )
            
        if self.pref_list is None: # Si no se especificaron prefijos, tomo como prefijos los nombres de las columnas + '_'
            self.pref_list = [ col+'_' for col in self.column_list ]
        
        minQty = self.minQty
        minPerc = self.minPerc
        maxCategs = len(data) if self.maxCategs == None else self.maxCategs
        otherPerc = self.otherPerc

        if self.verbose > 1:
            print('Columns', self.column_list)

        encoders   = {}
        drop_class = {}
        for col, pref in zip( self.column_list, self.pref_list ):

            labels = data[col].value_counts(ascending=True).rename('count').to_frame()
            labels['perc'] = 100 * labels['count'] / len(data)
            labels['cumPerc'] = labels['perc'].cumsum()
            labels = labels.reset_index().rename(columns={'index':'label'})
            cant_labels = len(labels)
            labels = labels[ (labels['count']   >= minQty) & 
                             (labels['perc']    >= minPerc) & 
                             (labels['cumPerc'] >  otherPerc ) ].head( maxCategs )
            encoders[col]   = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore').fit( labels['label'].to_frame() )
            # si no filtre ninguna categoria, marco la mayoritaria para dropear
            # en caso de haber filtrado categorias, no dropeo nada pues ceros en todas las columnas indica 'otros'
            drop_class[col] = labels.iloc[-1]['label'] if (cant_labels == len(labels)) else ''
            
        self.encoders   = encoders
        self.drop_class = drop_class

        return self
    
    def transform(self, X, y=None):

        if self.verbose > 0:
            print('LabelEncoder.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        
        if self.verbose > 1:
            print('Columns', self.column_list)

        data = pd.DataFrame(X) if type(X) is pd.Series else X # Si fue llamada con una serie, la transformo en DataFrame
        dummies = []

        for col, pref in zip( self.column_list, self.pref_list ): # para cada columna a transformar...

            dumm_names = [ pref + text2name( label ) for label in self.encoders[col].categories_[0] ]
            matrix = self.encoders[col].transform( data[col].to_frame() )
            dummies.append( pd.DataFrame(matrix, columns = dumm_names, index = data.index) )
            if self.drop_class[col]:
                dummies[-1].drop( columns = pref + text2name( self.drop_class[col] ), inplace=True )

        dummies = pd.concat(dummies, axis='columns')

        ret = pd.concat( [data, dummies], axis='columns' ).drop(columns=self.column_list)
        self.columnas_devueltas = list(ret.columns)
        return ret


class MultipleSelectionDummies(BaseEstimator, TransformerMixin):
    
    def __init__( self, delimiter = ';', column_list = None, pref_list = None, verbose = 0 ):
        """
        Crea un encoder que transforma la/s columna/s que contienen textos con combinaciones de seleccion de clases a una columna por clase valiendo 1 si está presente y 0 si no lo está.
        Ej: Tengo una columna de texto que vale 'rojo;amarillo;azul' o 'azul;verde;negro' o cualquier combinacion de entre opciones de colores que le gustan a un individuo. La transfomo
        en una dummie por cada color valiendo 1 si el color fue elegido y 0 si no lo fue. Esto difiere de las dummies tradicionales en que para ese caso tendriamos una columna por cada 
        COMBINACION en cambio con este encoder tenemos una columna por cada color individual.
        Parametros:
            delimiter: caracter que delimita las distintas opciones (default=';')
            column_list:  lista de columnas a transformar. None (default) para transformar todas las de tipo 'object' o 'category'
            pref_list: lista de prefijos a utilizar en cada columna. None (default) para usar el nombre de la columna de texto como prefijo. Las columnas dummies tendrán como nombre
                       el prefijo + el texto de la opcion (limpio de caracteres especiales)
            verbose: nivel de verbose (0: nada, 1: X.shape, 2: lo mismo mas el nombre de las columnas que procesa)
        """
        self.column_list = column_list
        self.pref_list = pref_list
        self.delimiter = delimiter
        self.verbose = verbose
    
    def fit( self, X, y=None ):

        if self.verbose > 0:
            print('MultipleSelectionDummies.fit - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        
        data = pd.DataFrame(X) if type(X) is pd.Series else X # Si fue llamada con una serie, la transformo en DataFrame

        if self.column_list is None: # Si no se especifico una lista de columnas, tomo todas las columnas de tipo object o category
            self.column_list = list( data.dtypes [ data.dtypes.apply( lambda x: str(x) in ['object','category'] )  ].index )
            
        if self.pref_list is None: # Si no se especificaron prefijos, tomo como prefijos los nombres de las columnas + '_'
            self.pref_list = [ col+'_' for col in self.column_list ]

        if self.verbose > 1:
            print('Columns', self.column_list)
            
        cols_fit = {} # estructura donde voy a guardar la info para hacer transform. Diccionario (key=columna) conteniendo lista de tuplas (nombre de la dummie, texto que le corresponde)
        for col, pref in zip( self.column_list, self.pref_list ):
            labels = set() # inicializo como conjunto (mas eficiente para eliminar repetidos)
            data[col].apply( lambda x: None if x is np.NaN else labels.update( set( x.split(self.delimiter) ) ) ) # para cada valor de la columna agrego al conjunto los textos nuevos
            labels = list(labels) # transformo el conjunto ya formado en una lista
            column_names = [ pref + text2name(label) for label in labels ] # obtengo los nombres que tendran las dummies (elimino caracteres especiales)

            cols_fit[col] = list( zip( column_names, labels ) ) # Agrego las tuplas a la estructura
            
        self.cols_fit = cols_fit # guardo la estructura en el objeto
        
        return self

    def transform(self, X, y=None):

        if self.verbose > 0:
            print('MultipleSelectionDummies.transform - X.shape', X.shape, '- y.shape', ('None' if y is None else y.shape))
        
        if self.verbose > 1:
            print('Columns', self.column_list)

        data = pd.DataFrame(X) if type(X) is pd.Series else X # Si fue llamada con una serie, la transformo en DataFrame
        dummies = pd.DataFrame()

        for column in self.cols_fit: # para cada columna a transformar...
            
            for dumm_name, dumm_text in self.cols_fit[column]: # para cada dummie a crear
                dummies[dumm_name] = data[column].str.contains(dumm_text, regex=False).astype(float) # creo la dummie

        ret = pd.concat( [data, dummies], axis='columns' ).drop(columns=self.column_list)
        self.columnas_devueltas = list(ret.columns)
        return ret
    

def corr(df, absValues=True, greater_than=None, less_than=None):
    """
    Devuelve la matriz de correlacion filtrada para que solo aparezcan las columnas que tienen alguna correlacion dentro del rango buscado
    
    Parametros:
    df: DataFrame a partir del cual se calcularan las correlaciones
    absValues: indica si se filtra por correlacion o por valor absoluto de la correlacion
    greater_than: devuelve solo columnas con alguna correlacion mayor que greater_than. Default (None) no filtra por mayor
    less_than: devuelve solo columnas con alguna correlacion menor que less_than. Default (None) no filtra por menor
    """
    if greater_than is None:
        greater_than = -2 # no filtro por greater_than
    if less_than is None:
        less_than = 2 # no filtro por less_than

    dfCorr = df.corr()
    cols = []

    for i in dfCorr.index:
        allButMe = dfCorr.loc[ dfCorr[i].index != i, i ] # serie de correlaciones con todas las columnas menos la i

        if absValues:
            allButMe = allButMe.abs()

        if ((allButMe > greater_than) & (allButMe < less_than)).sum() > 0: 
            cols.append(i)

    return df[cols].corr()



# wrapper para usar modelos de startsmodels desde funciones que esperan un formato compatible con sklearn 
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


class Modelo:
    """
    Clase para entrenar y testear modelos llevando un log de avances y almacenando data que permita operar comodamente con el.
    Ver parametros del metodo fit_test()
    """
    def __init__(self, **kwargs):
        self.log           = pd.DataFrame(columns=['trained_Rows', 'tested_Rows', 'num_Feats', 'cross_validation'])
        self.metricsTrain  = [('train_R2', metrics.r2_score, '{:.4f}')]
        self.metricsTest   = [('test_R2', metrics.r2_score, '{:.4f}')]
        self.estimator     = None
        self.scaler        = None
        self.sclFeats      = []
        self.notSclFeats   = []
        self.target        = ''
        self.cv            = None
        self.scoring       = 'accuracy'
        self._testStart_   = None
        if len(kwargs) > 0:
            self.fit_test( **kwargs )
        
        
    def __call__(self, **kwargs):
        """
        Equivalente a llamar el metodo fit_test(), ver parametros de esa funcion.
        """
        if len(kwargs) > 0:
            self.fit_test( **kwargs )


    def fit_test( self, trainData=None, testData=None, comment='', **kwargs ):
        """
        Entrena y almacena los datos de un modelo, lleva un log con las distintas pruebas que se fueron haciendo, guarda toda la info para re-entrenar o testear.

        trainData: DataFrame o lista de Series y DataFrames conteniendo la data de entrenamiento, X e y incluidas (None para no entrenar)
        testData: DataFrame o lista de Series y DataFrames conteniendo la data de test, X e y incluidas (None para no testear)
        
        Si no se especifica ni trainData ni testData pero si se especifican otros parametros (cv, feats, scaler, etc), estos de cambian en el objeto y quedan por defecto
        para proximas ejecuciones de fit_test

        kwargs:
        
            target: Nombre de la columna que será el target del modelo
            estimator: Objeto estimador que implementa el 'fit' con el cual vamos a entrenar y el 'predict' con el cual vamos a predecir
            scaler: objeto para estandarizar las variables
            metricsTrain: Lista de metricas a calcular para los datos de train y registrar en el log. Debe ser una lista de tuplas de 3 elementos:
                          1er elemento: nombre que tendra la columna en el log 
                          2do elemento: una funcion tal que funcion(y_true, y_pred) devuelva el valor de la metrica
                          3er elemento: (opcional) string de formato para el resultado de la metrica
                          default: [('train_R2', metrics.r2_score', {:.4f}')]
            metricsTest: idem anterior para las metricas que deban ejecutarse con los datos de test
                          default: [('test_R2', metrics.r2_score', {:.4f}')]
            
            feats: lista de la features del sum-modelo (si se deja vacia se debe especificar sclFeats o notSclFeats o ambas)
            sclFeats: Lista con los nombres de las columnas que serán las features numericas del sub-modelo (serán estandarizadas si se especifica un scaler)
            notSclFeats: Lista con los nombres de las columnas que serán las features del sub-modelo que no serán estandarizadas aunque se especifique un scaler
            
            cv: parametro cv para pasar a model_selection.cross_val_score(). None (default) para no hacer cross validation
            scoring: parametro scoring para pasar a model_selection.cross_val_score(). Default: 'accuracy'
            comment: comentario para agregar en el log (default=fecha-hora) (se mantiene un log con los resultados de las pruebas con fecha/hora mas el comentario adicional)
        
        Devuelve un objeto que contiene:
            model : el modelo entrenado
            scaler : el scaler entrenado
            trained : la cantidad de registros usados en el ultimo entrenamiento
            cvScores, meanScores, stdScores : la lista de scores obtenidos durante la cross validation, su media y desvio
            sclFeats, notSclFeats, target : las columnas de features y terget, para usar luego al hacer predict sobre el conjunto de datos que se desee
            log: un DataFrame con la historia de los resultados  obtenidos en sucesivos entrenamientos
        """

        # si no se informaron, usar lo almacenado en el objeto
        target       = self.target       if 'target'       not in kwargs else kwargs['target']
        estimator    = self.estimator    if 'estimator'    not in kwargs else kwargs['estimator']
        scaler       = self.scaler       if 'scaler'       not in kwargs else kwargs['scaler']
        cv           = self.cv           if 'cv'           not in kwargs else kwargs['cv']
        scoring      = self.scoring      if 'scoring'      not in kwargs else kwargs['scoring']
        metricsTrain = self.metricsTrain if 'metricsTrain' not in kwargs else kwargs['metricsTrain']
        metricsTest  = self.metricsTest  if 'metricsTest'  not in kwargs else kwargs['metricsTest']

        feats = [] if 'feats' not in kwargs else kwargs['feats']
        
        if 'feats' in kwargs:
            # voy a tener todas las features como escalables o no escalables, segun si tengo o no tengo scaler
            sclFeats    = []
            notSclFeats = []
        else:
            # voy a especificar las features individualmente segun si quiero que se escalen o no
            sclFeats    = self.sclFeats    if 'sclFeats'    not in kwargs else kwargs['sclFeats']
            notSclFeats = self.notSclFeats if 'notSclFeats' not in kwargs else kwargs['notSclFeats']
        
        # agregar feats a donde corresponda segun si hay o no hay scaler
        if scaler is None:
            notSclFeats = feats + notSclFeats
        else:
            sclFeats = feats + sclFeats

        # guardo en el objeto los posibles cambios de parametros
        self.sclFeats = sclFeats[:]
        self.notSclFeats = notSclFeats[:]
        self.target = target
        self.cv = cv
        self.scoring = scoring
        self.scaler = scaler
        self.estimator = estimator
        self.metricsTrain = metricsTrain[:]
        self.metricsTest = metricsTest[:]
        
        log = {'trained_Rows':'', 'tested_Rows':'', 'num_Feats':'', 'cross_validation':''}        

        if type(trainData) is list:
            trainData = pd.concat(trainData, axis='columns')

        if isinstance( trainData, pd.DataFrame ):
            rows = self.rows(trainData)
            log['trained_Rows'] = len(rows)
            log['num_Feats'] = len(sclFeats) + len(notSclFeats)
            self.trained = len(rows)

            x = self.x(trainData, rows=rows, fitScaler=True)
            y = self.y(trainData, rows=rows)
            
            cvScores = None
            if cv != None:
                cvScores = model_selection.cross_val_score( estimator, x, y, cv=cv, scoring=scoring )

            if cvScores is None:
                self.cvScores, self.meanScores, self.stdScores = None, None, None
            else:
                self.cvScores, self.meanScores, self.stdScores = cvScores, cvScores.mean(), cvScores.std()
                log['cross_validation'] = f"[{'{:.4f}'.format(cvScores.min())} - {'{:.4f}'.format(cvScores.max())}] x{len(cvScores)}"
                log['cross_validation'] += f"  ({'{:.4f}'.format(cvScores.mean())}, {'{:.4f}'.format(cvScores.std())})"

            self.model = estimator.fit( x, y )
            
            yPred = self.model.predict( x )
            for met in metricsTrain:
                m = met[1]( y, yPred ) # calculo la metrica
                f = '{}' if len(met) < 3 else met[2] # determino el formato
                c = met[0] if met[0] not in [t[0] for t in metricsTest] else met[0]+'_train' # determino el nombre de la columna de la metrica
                log[c] = f.format(m) # agrego la metrica al log

        if type(testData) is list:
            testData = pd.concat(testData, axis='columns')

        if isinstance( testData, pd.DataFrame ):
            rows = self.rows(testData)
            log['tested_Rows'] = len(rows)
            log['num_Feats'] = len(sclFeats) + len(notSclFeats)
            self.tested = len(rows)
            
            x = self.x(testData, rows=rows, fitScaler=False)
            y = self.y(testData, rows=rows)

            yPred = self.model.predict( x )
            for met in metricsTest:
                m = met[1]( y, yPred ) # calculo la metrica
                f = '{}' if len(met) < 3 else met[2] # determino el formato
                c = met[0] if met[0] not in [t[0] for t in metricsTrain] else met[0]+'_test' # determino el nombre de la columna de la metrica
                log[c] = f.format(m) # agrego la metrica al log

        if not comment:
            comment = datetime.now().strftime('%Y-%m-%d - %H:%M:%S')  # si no puso ningun comentario para el log, pongo fecha y hora...
            comment = comment + ' - Parameter(s) change' if ((trainData is None) and (testData is None)) else comment
        log['comment'] = comment

        self.log = self.log.append( log, ignore_index=True ) # agrego la fila nueva al log
        # reordeno las columnas para que comments quede al final y las posibles nuevas metricas justo antes
        self.log = pd.concat( [ self.log.drop(columns='comment'), self.log['comment'] ], axis='columns').fillna('')


    def predict( self, data, mode='t', rows=None, table=False, addResid=False, proba=False, threshold=None ):
        """
        Predice la columna target con el modelo entrenado.
        
        data: DataFrame o lista de Series y DataFrames conteniendo las features y el target del modelo
        mode: 't' (train o test) estima los valores de la columna target para todas las filas del modelo (feats y target no nulas)
              'i' (imputar) estima los valores de la columna target para las filas en que target es nulo, esta modalidad se
                  utiliza en sub-modelos entrenados para imputar una feature en funcion de otras del modelo principal
        rows: None (default) para procesar como indica el parametro mode o un pd.Index para especificar las filas a estimar
        table: False (default) para devolver una serie con los predichos, la serie tiene el nombre de target y en caso de mode='t' se le agrega el sufijo '_pred'
               True (solo para mode='t') para devolver una tabla de y_true vs y_pred. Las columnas tienen sufijos '_true' y '_pred' respectivamente
        resid: True (solo para table=True) para agregar una columna de residuos, esta columna tiene sufijo '_resid'
        proba: True si quiero devolver la probabilidad de la clase positiva (solo para modelos de clasificacion binaria, mas adelante generalizaré para multiclases)
        threshold: para modelos de clasificacion, si quiero devolver predicciones verdaderas solo para una probabilidad mayor al threshold
                   para devolver las probabilidades dejar en None
                   en regresiones, dejar en None

        Devuelve una Serie o Dataframe segun el parametro table, con las columnas pedidas y cuyos indices son los de las filas utilizadas para predecir
        """
        if type(data) is list:
            data = pd.concat(data,axis='columns')

        # corrijo posible incoherencia de parametros
        table = table and ( mode == 't' )
        addResid = addResid and table and (not proba) and (threshold is None)
        threshold = None if proba else threshold

        if isinstance( rows, pd.Index ):
            pass
        elif mode == 't':
            rows = self.rows(data)
        elif mode == 'i':
            rows = self.rows(data,'i')

        if proba or threshold != None:
            pred = self.model.predict_proba( self.x(data,rows=rows) )[:,1] # por ahora lo resuelvo para binarias, en un futuro generalizar...
        else:
            pred = self.model.predict( self.x(data,rows=rows) )

        if threshold != None:
            pred = (pred > threshold)

        pred = pd.Series( pred, index=rows, name = self.target + ('_pred' if mode=='t' else '') )
        
        cols = [ pred ]
        if table:
            true = self.y( data, rows ).rename(self.target+'_true')
            cols.append( true )
        
        if addResid:
            resid = ( true - pred ).rename(self.target+'_resid')
            cols.append( resid )
        
        return pred if len(cols) == 1 else pd.concat( cols, axis='columns' )


    def pop_log(self,*args): 
        """Deprecated, use logPop"""
        self.logPop(*args) 
        
    def logPop(self, n=1):
        """
        Borra los ultimos n registros del log
        """
        self.log.drop(index = self.log.tail(n).index, inplace = True)

    def logTestStart(self):
        """
        Registra la linea del log a partir de la cual se va a comenzar a hacer un test
        Pensado para hacer fiteos de prueba que luego se borraran del log con el metodo logTestEnd()
        """
        self._testStart_ = len(self.log)

    def logTestEnd(self):
        """
        Devuelve un pd.DataFrame todas las lineas del log generadas desde el ultimo logTestStart() y las borra del log principal
        """
        if self._testStart_ is None:
            return pd.DataFrame()
        ret      = self.log.iloc[ self._testStart_ : ]
        self.log = self.log.iloc[ : self._testStart_ ]
        self._testStart_ = None
        return ret 

    def logAddData(self, column, value):
        """
        Agrega datos en una columna (existente o nueva), en la ultima linea del log (la generada mas recientemente)
        
        column: columna donde agregar el dato
        value: el dato a agregar
        """
        self.log.loc[ len(self.log)-1, column ] = value
        self.log = self.log.fillna('')

    def rows(self, data, mode='t'):
        """
        Devuelve el indice de las filas en que las columnas del modelo no tienen ningun nulo
        data: DataFrame o lista de Series y DataFrames conteniendo las features y el target del modelo
        mode: 't' (train o test) devuelve el indice de todas las filas del modelo (feats y target no nulas)
              'i' (imputar) devuelve el indice de las filas en que las features son no nulas pero el target es nulo, esta modalidad se
                  utiliza en sub-modelos entrenados para imputar una feature en funcion de otras del modelo principal
        """
        if type(data) is list:
            data = pd.concat(data,axis='columns')

        if mode=='t':
            ret = data[ self.sclFeats + self.notSclFeats + [self.target] ].dropna(axis='index', how='any').index
        else:
            ret = data[ self.sclFeats + self.notSclFeats ][ data[self.target].isnull() ].dropna(axis='index', how='any').index

        return ret


    def cols(self):
        """
        Devuelve una lista con las columnas del modelo
        """
        return self.sclFeats + self.notSclFeats


    def x(self, data, noScale=False, fitScaler=False, rows=None):
        """
        Devuelve un pd.DataFrame con los valores de x que se usarían en el modelo si se entrenara/testeara con la data pasada por parametro

        data: DataFrame o lista de Series y DataFrames conteniendo las features y el target del modelo
        noScale: False (default) para escalar segun lo indique el modelo, True no escalar independientemente de lo que diga el modelo
        fitScaler: False (default) para escalar sin re-entrenar en scaler (transform), True para re-entrenar (fit_transform)
        rows: None (default) para devolver las filas del modelo que no contengan nulos en features ni target, 
              si se especifica, devuelve las filas pedidas (debe ser un pd.Index)

        Toma todas las columnas del modelo y todas las filas que correspondan segun el parametro rows 
        """
        if type(data) is list:
            data = pd.concat(data,axis='columns')

        if rows is None:
            rows = self.rows(data)

        if self.scaler is None or noScale:
            x = data.loc[rows,self.sclFeats].values
        else:
            if fitScaler:
                x = self.scaler.fit_transform( data.loc[rows,self.sclFeats].values )
            else:
                x = self.scaler.transform( data.loc[rows,self.sclFeats].values )

        x = np.concatenate( ( x, data.loc[rows,self.notSclFeats].values ), axis=1 )
        
        return pd.DataFrame( x, index=rows, columns=self.sclFeats+self.notSclFeats )



    def y(self, data, rows=None):
        """
        Devuelve una pd.Series con los valores de y que se usarían en el modelo si se entrenara/testeara con la data pasada por parametro

        data: DataFrame o lista de Series y DataFrames conteniendo las features y el target del modelo
        rows: None (default) para devolver las filas del modelo que no contengan nulos en features ni target, 
              si se especifica, devuelve las filas pedidas (debe ser un pd.Index)

        Toma las filas que corresponda segun el parametro rows y devuelve una serie con la columna target 
        """
        if type(data) is list:
            data = pd.concat(data,axis='columns')

        if rows is None:
            rows = self.rows(data)

        return data.loc[ rows, self.target ]


    def df(self,data):
        """
        Devuelve un DataFrame con la info que se utilizaría si se entrenara/testeara el modelo con la data pasada por parametro 

        data: DataFrame o lista de Series y DataFrames conteniendo las features y el target del modelo
        """
        if type(data) is list:
            data = pd.concat(data,axis='columns')

        return data[self.cols()].dropna(axis='index', how='any')


    def __str__(self):
        return f'trained ({self.trained},{len(self.sclFeats)}+{len(self.notSclFeats)}), ' + \
               f'cvScores: {str(self.cvScores.round(4))}, Mean: {self.meanScores.round(4)}, Std: {self.stdScores.round(4)}'



def MAPE( y_true, y_pred ):
    """
    (provisorio)
    Calculo esta metrica que por ahora no está en la ultima version que tengo de sklearn
    mean absolute precentage error, eror medio porcentual
    """
    mape = np.mean( np.abs( (y_true - y_pred) / y_true ) ) * 100
    return mape


def specificity_score( true, pred ):
    """
    Provisorio hasta que sklearn incluya esta metrica (si lo hace)
    """
    conf = metrics.confusion_matrix(true,pred)
    return conf[0,0] / ( conf[0,0] + conf[0,1] )


def class_metrics(true, pred_proba, threshold=None, mets='aprsfA' , **kwargs):
    """
    Devuelve metrics.recall_score() segun los parametros dados. Permite buscar valor de threshold para fijar un valor de precision

    true: pd.Series o array conteniendo los valores verdaderos
    pred_proba: pd.Series o array conteniendo la probabilidad predicha por el modelo
    threshold: trheshold para determinar el valor predicho (predicho = pred_proba > threshold). Si es None (default), threshold = 0.5

    recall: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a este recall. Devolvera todas las metricas pedidas segun ese threshold calculado.
    precision: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a esta precision. Devolvera todas las metricas pedidas segun ese threshold calculado.
    specificity: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a esta specificity. Devolvera todas las metricas pedidas segun ese threshold calculado.

    mets: string especificando las metricas que se quieren devolver (a:accuracy, f:f1_score, r:recall, p:precision, s:specificity, A:roc_auc, c:confusion_matrix, t:threshold calculado/especificado)
    
    Devuelve una tupla con todas las metricas solicitadas, por orden de aparicion en mets. Si se pidio una sola metrica, devuelve un float
    """
    maxIter = 16
    metricFunc = dict(r = metrics.recall_score, 
                      p = metrics.precision_score, 
                      s = specificity_score, 
                      a = metrics.accuracy_score, 
                      f = metrics.f1_score, 
                      A = metrics.roc_auc_score,
                      c = metrics.confusion_matrix
                    )

    fixed = False # indica si se fijo un valor a recall, precision o specificity y por lo tanto debo calcular el threshold correspondiente
    if 'recall' in kwargs:
        fixed, fixValue, fixFunc, fixAscends = True, kwargs['recall'], metricFunc['r'], False
    elif 'precision' in kwargs:
        fixed, fixValue, fixFunc, fixAscends = True, kwargs['precision'], metricFunc['p'], True
    elif 'specificity' in kwargs:
        fixed, fixValue, fixFunc, fixAscends = True, kwargs['specificity'], metricFunc['s'], True
    
    if fixed:
        # busqueda dicotomica (asumo que las curvas sobre las que estoy buscando son siempre ascendentes o siempre descendentes, 
        # quizá podría mejorar este parrafo usando un optimizador tipo scipy.optimize.minimize)
        inf, thr, sup = 0, 0.5, 1
        i = 0
        while True:
            current = fixFunc( true, (pred_proba > thr).astype(int) )
            i+=1
            if current == fixValue or i > maxIter: # encontre o pase el limite de iteraciones
                break
            if (current < fixValue) == fixAscends: # Si estoy a la izquierda del valor buscado y la curva es ascendente (o viceversa)
                inf, thr, sup = thr, (thr+sup)/2, sup # me desplazo hacia la derecha
            else:
                inf, thr, sup = inf, (inf+thr)/2, thr # me desplazo hacia la izquierda
    else:
        thr = 0.5 if threshold is None else threshold
        
    ret = []
    for m in mets:
        if m == 't':
            ret.append(thr)
        elif m == 'A': # auc es el unico que se calcula con pred_proba !!
            ret.append( metricFunc[m]( true, pred_proba ) )
        else:
            ret.append( metricFunc[m]( true, (pred_proba > thr).astype(int) ) )
    
    return ret[0] if len(ret) == 1 else tuple(ret) 


def class_graph( true, pred_proba, threshold=None, mets='rp', figsize=(15,10), **kwargs ):
    """
    Grafica la curva ROC, la matriz de confusion y una o varias metricas en funcion del threshold
    
    true: array o pd.Series con los valores verdaderos de y
    pred_proba: array o pd.Series con los valores de probabilidad predicha
    

    recall: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a este recall. Informa el resto de las metricas que se corresponden con ese threshold
    precision: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a esta precision. Informa el resto de las metricas que se corresponden con ese threshold
    specificity: (opcional) si se especifica, ignora el parametro threshold y en lugar de eso calcula el treshold correspondiente a esta specificity. Informa el resto de las metricas que se corresponden con ese threshold

    mets: string especificando las metricas que se quieren graficar (a:accuracy, f:f1_score, r:recall, p:precision, s:specificity)
    """
    
    # calculo todas las metricas
    thr, acc, pre, rec, spe, f1, auc, mConf = class_metrics( true, pred_proba, threshold = threshold, mets = 'taprsfAc', **kwargs )

    # Calculo la curva ROC
    fpr_log,tpr_log,thr_log = metrics.roc_curve( true, pred_proba )
    roc = pd.DataFrame( dict(fpr = fpr_log, tpr = tpr_log, thr = thr_log) )

    # calculo accuracy y F1 para threshold = 0.5 pues en el grafico de curva ROC muestro esos valores estandar...
    acc_05 = metrics.accuracy_score(true, (pred_proba > 0.5))
    f1_05  = metrics.f1_score(true, (pred_proba > 0.5))

    # calculo la curva de metricas vs threshold
    thr_log = np.arange(0,1, step = 1/250) # genero 250 puntos
    conf = np.array([ metrics.confusion_matrix( true, (pred_proba > t) ) for t in thr_log ] )
    tn, fp, fn, tp = conf[:,0,0], conf[:,0,1], conf[:,1,0], conf[:,1,1]
    acc_log = ( tp + tn ) / ( tp + tn + fp + fn )
    pre_log = tp / ( tp + fp )
    rec_log = tp / ( tp + fn )
    spe_log = tn / ( tn + fp )
    f1_log  = tp*2 / ( tp*2 + fp + fn )
    rpt = pd.DataFrame( dict(thr = thr_log, acc = acc_log, pre = pre_log, rec = rec_log, spe = spe_log, f1 = f1_log ) )

    # grafico
    fig, ((axRoc, axMT), (axConf,axRP) ) = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 3]} )

    # grafico ROC
    axRoc.axis([0, 1.01, 0, 1.01])
    axRoc.set_xlabel('1 - Specificty'); axRoc.set_ylabel('TPR / Sensitivity'); axRoc.set_title('ROC Curve')
    axRoc.plot(roc.fpr,roc.tpr)
    axRoc.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01), color='grey')
    axRoc.annotate(f'AUC = {round(auc,5)}\nF1 = {round(f1_05,5)}\nAccuracy = {round(acc_05,5)}\n(Thr=0.5)', (0.55,0.2))

    # grafico Metrics vs Threshold
    legend, annot = [], ''
    metField = dict( a='acc', p='pre', r='rec', s='spe', f='f1' )
    metDesc  = dict( a='Accuracy', p='Precision', r='Recall', s='Specificity', f='F1 Score' )
    for m in mets:
        sns.lineplot(x = rpt['thr'], y=rpt[ metField[m] ], ax=axMT)
        legend.append(metDesc[m])
        annot = annot + '\n' if annot != '' else ''
        annot = annot + f'{metDesc[m]} = {{{metField[m]}}}'
    if annot != '':
        annot = annot + '\n' + 'Threshold = {thr}'
    axMT.legend( legend )
    axMT.set_xlabel('Threshold'); axMT.set_ylabel('Metric value'); axMT.set_title('Metrics vs Threshold')
    if threshold is not None or any(param in kwargs for param in ['recall','precision','specificity']): # si especifico un threshold o pidio fijar alguna metrica, muestro el resultado de esa operacion
        axMT.vlines( thr, 0, 1, color='grey' )
        posicX = thr+0.03 if thr < 0.5 else thr-0.35
        axMT.annotate( annot.format(thr=round(thr,5), acc=round(acc,5), pre=round(pre,5), rec=round(rec,5), spe=round(spe,5), f1=round(f1,5)), (posicX,0.065+len(legend)*0.065) )

    # grafico Precision vs Recall
    sns.lineplot(x = rpt['rec'], y=rpt['pre'], ax=axRP)
    axRP.set_xlabel('Recall'); axRP.set_ylabel('Precision'); axRP.set_title('Precision vs Recall')


    # grafico matriz de confusion
    mConf = pd.DataFrame( data = mConf, index = pd.Series(true).unique(), columns = pd.Series(true).unique() )    
    sns.heatmap(mConf, fmt='d', annot=True, cmap='YlGnBu', ax=axConf)
    axConf.set_xlabel(f'Predicted\n(Threshold={round(thr,5)})'); axConf.set_ylabel('Actual'); axConf.set_title('Confusion Matrix')

    plt.show()

