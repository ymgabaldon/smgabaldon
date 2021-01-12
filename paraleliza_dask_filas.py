import multiprocessing as mp
import dask.dataframe as dd

def paraleliza_dask_filas(pd_df,funcion,n_cores=0,scheduler="processes"):
    """Función para paralelizar funciones que se aplican en dataframes fila a fila típicas de aplicación con lambdas.
    La idea es complementar a swifter que por algún motivo, no permite aplicarse con funciones de texto.
    Revisar porque no consigo que me funcione con funciones sencillas, si bien para temas de textos sí mejoró
    Argumentos:
    pd_df: Columna de un Dataframe a la que se le aplica la función;
    funcion: La función a aplicar al dataframe y que se aplica en la lambda;
    n_cores es una variable para definir el número de particiones de acuerdo con la fórmula 2*n_cores. Si no se especifica,
    se tomarán todos los cores;
    scheduler: "planificador" que utilizará dask
    Devuelve una serie con el resultado de aplicar la función a la columna del dataframe"""
    if n_cores==0:
        n_cores=mp.cpu_count()
    ### Creamos las particiones del dataframe  
    dask_df=dd.from_pandas(pd_df,npartitions=2*n_cores)
    ###Aplicamos la función a cada una de las particiones:
    salida=dask_df.map_partitions(lambda dataframe: dataframe.apply((lambda fila: funcion(fila)),axis=1)).compute(scheduler=scheduler)
    return salida
    