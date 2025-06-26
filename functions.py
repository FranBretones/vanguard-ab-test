import pandas as pd # Librería para análisis de datos
import numpy as np # Librería para análisis numérico
from scipy import stats # Librería para estadísticas
from statsmodels.stats.proportion import proportions_ztest # Librería para prueba de hipótesis
import matplotlib.pyplot as plt # Librería para graficos
import seaborn as sns # Librería para graficos

# Funciones de limpieza y preprocesamiento de datos

def limpieza_demograficos(df):
    """
    Limpia y transforma el dataset demográfico aplicando varias reglas:

    Descripción de pasos:
    ---------------------
    1. Elimina registros duplicados y filas con valores faltantes.
    2. Convierte ciertas columnas a tipo entero ('clnt_tenure_yr', 'clnt_age', 'num_accts', 'logons_6_mnth', 'calls_6_mnth') y reemplaza 'X' por 'U' en la columna 'gendr'.
    3. Filtra los datos para conservar solo clientes mayores de 18 años y con menos de 40 años de permanencia.
    4. Clasifica la columna 'bal' en categorías: 'bajo', 'medio' y 'alto'.
    5. Elimina registros donde la permanencia es mayor que la edad del cliente.
    6. Segmenta los clientes en 'new' o 'long_tenure' según los años de permanencia.
    7. Agrupa a los clientes por edad en segmentos usando el rango intercuartílico (IQR).
    8. Selecciona únicamente las columnas necesarias para el análisis final.

    Parámetros:
    -----------
    df : DataFrame
        Datos demográficos originales.

    Retorna:
    --------
    DataFrame limpio y transformado, listo para análisis.
    """
    # 1
    df = df.drop_duplicates()
    df = df.dropna()

    # 2
    df[["clnt_tenure_yr","clnt_age","num_accts","logons_6_mnth","calls_6_mnth"]] = df[["clnt_tenure_yr","clnt_age","num_accts","logons_6_mnth","calls_6_mnth"]].astype(int)
    df["gendr"] = df["gendr"].replace({"X": "U"})

    # 3
    df = df[(df["clnt_age"] >= 18) & (df["clnt_tenure_yr"] < 40)]

    # 4
    # Clasificación de balances en categorías usando cuantiles
    df['bal_category'] = pd.qcut(df['bal'], q=[0, 0.25, 0.75, 1], labels=['bajo', 'medio', 'alto'])

    # 5
    indices_a_eliminar = df[df["clnt_tenure_yr"] > df["clnt_age"]].index
    df = df.drop(indices_a_eliminar)

    # 6
    df['tenure_segment'] = np.where(df['clnt_tenure_yr'] < 10, 'new', 'long_tenure')

    # 7 - Segmentación por edad usando IQR
    Q1 = df['clnt_age'].quantile(0.25)
    Q3 = df['clnt_age'].quantile(0.75)
    IQR = Q3 - Q1

    def age_iqr_segment(age):
        if age < Q1:
            return 'young'
        elif age > Q3:
            return 'old'
        else:
            return 'middle'

    df['age_segment'] = df['clnt_age'].apply(age_iqr_segment)

    # 8
    df = df[["client_id", "clnt_tenure_yr", "clnt_age", "gendr", "num_accts", "bal", "calls_6_mnth", "logons_6_mnth", "tenure_segment", "age_segment",'bal_category']]

    return df

def limpieza_experiment(df):
    """
    Limpia los datos del proceso experimental de clientes.

    Operaciones realizadas:
    -----------------------
    1. Eliminar valores nulos:
    - Se eliminan las filas que contienen valores nulos en cualquier columna del DataFrame.
    - La accion corresponde a trabajar solo con clientes que pertenecieron a un grupo control o test.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos de clientes experimentales.

    Returns:
    --------
    pd.DataFrame
        DataFrame limpio sin valores nulos.
    """
    # 1
    df = df.dropna()

    return df

def limpieza_webdata(df1, df2):
    """
    hace un merge de los dos DataFrames de datos web y aplica transformaciones para su limpieza y análisis.

    Descripción :
    -------------------------------
    1. Combina ambos DataFrames en uno solo.
    2. Convierte la columna 'date_time' al tipo datetime.
    3. Ordena los datos por 'client_id', 'visit_id' y 'date_time'.
    4. Genera la columna 'is_complete' para marcar procesos finalizados.
    5. Calcula el tiempo transcurrido entre cada paso ('time_diff_step').
    6. Establece el orden lógico de los pasos del proceso.
    7. Identifica el paso anterior y detecta posibles errores en la secuencia.

    Parámetros:
    -----------
    df1, df2 : pd.DataFrame
        DataFrames con los datos web a combinar.

    Retorna:
    --------
    pd.DataFrame
        DataFrame combinado y transformado listo para análisis.
    """
    # 1
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 2
    merged_df["date_time"] = pd.to_datetime(merged_df["date_time"])

    # 3
    merged_df = merged_df.sort_values(by=['client_id', 'visit_id', 'date_time'])

    # 4
    final_steps = ['confirm']
    merged_df['is_complete'] = merged_df.groupby('client_id')['process_step'].transform(lambda x: 'Complete' if final_steps[0] in x.values else 'Incomplete')
    merged_df['is_complete'] = merged_df['is_complete'].replace({'Complete': 1, 'Incomplete': 0})

    # 5
    merged_df['time_before_step'] = merged_df.groupby(['client_id', 'visitor_id', 'visit_id'])['date_time'].shift(1)
    merged_df['time_diff_step'] = (merged_df['date_time'] - merged_df['time_before_step']).dt.total_seconds()

    # 6
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']
    merged_df['process_step'] = pd.Categorical(merged_df['process_step'], categories=step_order, ordered=True)
    merged_df['prev_process_step'] = merged_df.groupby(['client_id', 'visit_id'])['process_step'].shift(1)

    # 7
    merged_df['is_error'] = merged_df['process_step'] < merged_df['prev_process_step']
    merged_df['is_error'] = merged_df['is_error'].fillna(False)


    return merged_df

# ----------------------------------------------------- #
# Funciones para las visualizaciones 
# ----------------------------------------------------- #

def plot_variables_numericas(df, cols):
    """
    Grafica la distribución de variables numéricas usando histogramas con KDE.

    Parameters:
        df (DataFrame): DataFrame de entrada.
        cols (list): Lista de nombres de columnas numéricas a graficar.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    for idx, col in enumerate(cols, 1):
        plt.subplot(3, 2, idx)
        sns.histplot(df[col], bins=100, kde=True, color='steelblue')
        plt.title(col.replace('_', ' ').title(), fontsize=11)
        plt.xlabel('')
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.suptitle('Distribución de variables numéricas', fontsize=14, y=1.02)
    plt.show()
    
    

def outliers_boxplots(df):
    """
    Genera caja de bigotes para cada columna numérica en el DataFrame proporcionado.
    
    Parameters:
        df (DataFrame): DataFrame que contiene las columnas numéricas.
    """

    # Configurar el tamaño de la figura
    num_columns = df.shape[1]
    num_rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calcular número de filas necesarias
    plt.figure(figsize=(10, num_rows * 4))  # Ajustar el tamaño en función del número de filas

    # Iterar sobre cada columna y generar un boxplot
    for i, column in enumerate(df.columns):
        plt.subplot(num_rows, 3, i + 1)  # Cambia el tamaño de la cuadrícula según el número de columnas
        plt.boxplot(df[column].dropna())  # Elimina valores NaN antes de graficar
        plt.title(column)
        plt.grid(True)

    plt.tight_layout()  # Ajustar el espaciado entre los gráficos
    plt.suptitle('Boxplots de Variables Numéricas', y=1.02)  # Ajustar el título para que no se superponga
    plt.show()
    

def plot_age_distribution_by_segment(df, age_col='clnt_age', segment_col='age_segment'):
    """
    Grafica la distribución de edad por segmento usando un histograma.

    Parámetros:
        df (DataFrame): DataFrame de entrada.
        age_col (str): Nombre de la columna de edad.
        segment_col (str): Nombre de la columna de segmento de edad.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=age_col,
        bins=100,
        kde=False,
        color="steelblue",
        hue=segment_col,
        multiple="stack",
        palette="Set2"
    )
    plt.title('Distribución de edad por segmento')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()
    

def plot_tenure_segment_distribution(df, tenure_col='clnt_tenure_yr', segment_col='tenure_segment'):
    """
    Grafica la distribución de clientes por segmento de tenencia y añade etiquetas con el total de cada grupo.

    Parámetros:
        df (DataFrame): DataFrame de entrada que debe contener las columnas de tenencia y segmento.
        tenure_col (str): Nombre de la columna de años de tenencia.
        segment_col (str): Nombre de la columna de segmento de tenencia.
    """
    tenure_frequency_table = df[segment_col].value_counts()
    
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df,
        x=tenure_col,
        bins=100,
        kde=False,
        color="steelblue",
        hue=segment_col,
        multiple="stack",
        palette="Set2"
    )

    # Añadir etiquetas con el total de cada grupo
    for i, segment in enumerate(tenure_frequency_table.index):
        total = tenure_frequency_table[segment]
        plt.text(
            x=0.95,
            y=0.95 - i*0.08,
            s=f"{segment}: {total:,}",
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=12, color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    plt.title('Distribución de clientes por segmento de tenencia')
    plt.xlabel('Años de permanencia')
    plt.ylabel('Frecuencia')
    plt.show()
    
def analizar_tenencia_segmentada(df, col_tenencia='clnt_tenure_yr', col_segmento='tenure_segment'):
    """
    Analiza y grafica la distribución de la tenencia de clientes con las barras coloreadas por segmento.
    Se muestran líneas de Q1, Q3 y límites del IQR.
    """
    # Cálculo de cuartiles e IQR
    Q1 = df[col_tenencia].quantile(0.25)
    Q3 = df[col_tenencia].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = max(Q1 - 1.5 * IQR, 0)
    upper_limit = Q3 + 1.5 * IQR

    # Frecuencia por segmento
    tenure_freq = df[col_segmento].value_counts()

    # Gráfico
    plt.figure(figsize=(12, 6))
    ax = sns.histplot(
        data=df,
        x=col_tenencia,
        bins=50,
        hue=col_segmento,
        multiple="stack",
        palette="tab10",
        kde=False
    )

    # Líneas de referencia
    plt.axvline(Q1, color='red', linestyle='--', label='Q1')
    plt.axvline(Q3, color='green', linestyle='--', label='Q3')
    plt.axvline(lower_limit, color='orange', linestyle='--', label='Límite inferior')
    plt.axvline(upper_limit, color='purple', linestyle='--', label='Límite superior')

    # Títulos y etiquetas
    plt.title('Distribución de años de tenencia (segmentada)')
    plt.xlabel('Años de tenencia')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def analizar_rango_tenencia(df, tenure_col='clnt_tenure_yr', client_id_col='client_id', bins=100):
    """
    Analiza el rango intercuartil (IQR) de la tenencia y grafica su distribución.

    Parámetros:
        df (DataFrame): DataFrame de entrada.
        tenure_col (str): Nombre de la columna de años de tenencia.
        client_id_col (str): Nombre de la columna de identificador de cliente.
        bins (int): Número de bins para el histograma.

    Retorna:
        dict: Estadísticas calculadas (Q1, Q3, IQR, límites y conteos).
    """
    Q1 = df[tenure_col].quantile(0.25)
    Q3 = df[tenure_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = max(Q1 - 1.5 * IQR, 0)
    upper_limit = Q3 + 1.5 * IQR

    num_clientes = df[client_id_col].nunique()
    total = len(df)
    dentro_limites = df[(df[tenure_col] >= lower_limit) & (df[tenure_col] <= upper_limit)].shape[0]
    debajo_limite = df[df[tenure_col] < lower_limit].shape[0]
    encima_limite = df[df[tenure_col] > upper_limit].shape[0]

    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Límite inferior (ajustado): {lower_limit}, Límite superior (Q3 + 1.5 * IQR): {upper_limit}")
    print(f"Cantidad de clientes únicos: {num_clientes}")
    print(f"Total registros: {total}")
    print(f"Registros dentro de los límites: {dentro_limites}")
    print(f"Registros debajo del límite inferior: {debajo_limite}")
    print(f"Registros encima del límite superior: {encima_limite}")

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=tenure_col,
        bins=bins,
        kde=True,
        color="steelblue"
    )
    plt.title('Distribución de años de tenencia')
    plt.xlabel('Años de tenencia')
    plt.ylabel('Frecuencia')
    plt.axvline(Q1, color='red', linestyle='--', label='Q1')
    plt.axvline(Q3, color='green', linestyle='--', label='Q3')
    plt.axvline(lower_limit, color='orange', linestyle='--', label='Límite inferior (ajustado)')
    plt.axvline(upper_limit, color='purple', linestyle='--', label='Límite superior (Q3 + 1.5 * IQR)')
    plt.legend()
    plt.show()

    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'num_clientes': num_clientes,
        'total': total,
        'dentro_limites': dentro_limites,
        'debajo_limite': debajo_limite,
        'encima_limite': encima_limite
    }
    

# --------------------------------------------------------- #
# Funciones de KPIS
# --------------------------------------------------------- #

def estadisticas_completion_rate(df):
    """
    Calcula las tasas de finalización e incompletación por cliente único,
    usando solo la última interacción de cada cliente.
    """
    # Nos quedamos con el último registro por cliente
    df = df.sort_values('date_time').drop_duplicates('client_id', keep='last')

    results = []
    for variation in ['Control', 'Test']:
        filtered_df = df[df["Variation"] == variation]
        
        completed = (filtered_df['is_complete'] == 1).sum()
        incomplete = (filtered_df['is_complete'] == 0).sum()
        total = completed + incomplete

        results.append({
            'Variation': variation,
            'Completed Clients': completed,
            'Completion Rate (%)': (completed / total) * 100 if total else 0,
        })

    return pd.DataFrame(results)

def calculo_tiempo_pasos_stats(df):
    """
    Calcula estadísticas de tiempo por paso de proceso y por grupo (Control y Test).
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información del tiempo por paso.
        
    Returns:
        DataFrame: Estadísticas de tiempo para cada paso, desglosadas por grupo.
    """
    # Agrupar por 'Variation' y 'process_step' y calcular estadísticas de tiempo
    time_stats_df = df.groupby(['Variation', 'process_step'])['time_diff_step'].agg(['mean', 'min', 'max', 'count']).rename(
        columns={'mean': 'mean_time', 'min': 'min_time', 'max': 'max_time', 'count': 'num_entries'}
    ).reset_index()
    
    return time_stats_df

def medias_tiempo_segmento(df, column):
    # Calcular el tiempo total por visita
    visit_times = df.groupby('visit_id')['time_diff_step'].sum().reset_index()
    visit_times.rename(columns={'time_diff_step': 'time_per_visit_in_sec'}, inplace=True)
    
    # Unir info por visita 
    visit_info = df.drop_duplicates(subset='visit_id')[['visit_id', 'Variation', 'age_segment', 'tenure_segment']]
    visit_data = pd.merge(visit_info, visit_times, on='visit_id', how='left')

    # Inicializar listas
    variations = visit_data[column].unique()
    age_segment_results = []
    tenure_segment_results = []

    for variation in variations:
        filtered_df = visit_data[visit_data[column] == variation]

        # Promedio por segmento de edad
        age_avg = filtered_df.groupby('age_segment')['time_per_visit_in_sec'].mean().reset_index()
        age_avg['Variation'] = variation
        age_avg.rename(columns={'time_per_visit_in_sec': 'Average Time (seconds)'}, inplace=True)
        age_segment_results.append(age_avg)

        # Promedio por segmento de tenure
        tenure_avg = filtered_df.groupby('tenure_segment')['time_per_visit_in_sec'].mean().reset_index()
        tenure_avg['Variation'] = variation
        tenure_avg.rename(columns={'time_per_visit_in_sec': 'Average Time (seconds)'}, inplace=True)
        tenure_segment_results.append(tenure_avg)

    # Concatenar resultados
    df_age_segment_results = pd.concat(age_segment_results).reset_index(drop=True)
    df_tenure_segment_results = pd.concat(tenure_segment_results).reset_index(drop=True)

    # Exportar
    df_age_segment_results.to_csv('data/visualization/average_time_by_age_segment.csv', index=False)
    df_tenure_segment_results.to_csv('data/visualization/average_time_by_tenure_segment.csv', index=False)

    return df_age_segment_results, df_tenure_segment_results

def error_rate_calculo(df):
    """
    Calcula la tasa de errores en el DataFrame proporcionado para los grupos Control y Test.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y el grupo (Control o Test).
        
    Returns:
        DataFrame: Un DataFrame con la tasa de errores y el total de registros y errores para ambos grupos.
    """
    # Función interna para calcular la tasa de errores
    def error_stats(group_df):
        total_records = len(group_df)
        total_errors = group_df['is_error'].sum()
        error_rate = (total_errors / total_records) * 100 if total_records > 0 else 0
        return total_records, total_errors, round(error_rate, 2)

    # Calcular estadísticas para el grupo Control
    control_df = df[df['Variation'] == 'Control']
    control_stats = error_stats(control_df)
    
    # Calcular estadísticas para el grupo Test
    test_df = df[df['Variation'] == 'Test']
    test_stats = error_stats(test_df)

    # Crear un DataFrame para devolver los resultados
    result_df = pd.DataFrame({
        'Group': ['Control', 'Test'],
        'Total Records': [control_stats[0], test_stats[0]],
        'Total Errors': [control_stats[1], test_stats[1]],
        'Error Rate (%)': [control_stats[2], test_stats[2]]
    })

    return result_df

# error rate por segmento de edad
def error_rate_segmento_edad_calc(df):
    """
    Calcula la tasa de errores por segmento de edad en el DataFrame proporcionado.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y segmentos de edad.
        
    Returns:
        DataFrame: Un DataFrame con la tasa de errores por segmento de edad.
    """
    # Agrupar por 'age_segment' y calcular estadísticas de errores
    error_stats = df.groupby('age_segment')['is_error'].agg(['count', 'sum']).reset_index()
    error_stats.rename(columns={'count': 'Total Records', 'sum': 'Total Errors'}, inplace=True)
    
    # Calcular la tasa de errores
    error_stats['Error Rate (%)'] = (error_stats['Total Errors'] / error_stats['Total Records']) * 100
    error_stats['Error Rate (%)'] = error_stats['Error Rate (%)'].round(2)
    
    return error_stats

def calculate_error_rate_by_tenure_segment(df):
    """
    Calcula la tasa de errores por segmento de tenencia en el DataFrame proporcionado.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y segmentos de tenencia.
        
    Returns:
        DataFrame: Un DataFrame con la tasa de errores por segmento de tenencia.
    """
    # Agrupar por 'tenure_segment' y calcular estadísticas de errores
    error_stats = df.groupby('tenure_segment')['is_error'].agg(['count', 'sum']).reset_index()
    error_stats.rename(columns={'count': 'Total Records', 'sum': 'Total Errors'}, inplace=True)
    
    # Calcular la tasa de errores
    error_stats['Error Rate (%)'] = (error_stats['Total Errors'] / error_stats['Total Records']) * 100
    error_stats['Error Rate (%)'] = error_stats['Error Rate (%)'].round(2)
    
    return error_stats

# --------------------------------------------------------- #
# Funciones pruebas de hipótesis
# --------------------------------------------------------- #

def hypothesis_testing_tasa_finalizacion(df, alpha=0.05):
    
    """ Realiza una prueba de hipótesis para comparar las tasas de finalización entre los grupos Test y Control.
    Parámetros:
        df (DataFrame): DataFrame que contiene los datos de clientes y sus pasos de proceso.
        alpha (float): Nivel de significancia para la prueba (por defecto 0.05).
    Retorna:
        dict: Un diccionario con el estadístico z, el valor p, la decisión de la prueba y las tasas de finalización.
    """
    
    # Asegurar que las columnas necesarias estén presentes
    required_columns = {'Variation', 'client_id', 'process_step'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Faltan columnas requeridas: {required_columns - set(df.columns)}")
    
    # Obtener clientes únicos por grupo
    test_df = df[df['Variation'] == 'Test']
    control_df = df[df['Variation'] == 'Control']
    
    # Clientes únicos por grupo
    unique_test_clients = test_df['client_id'].nunique()
    unique_control_clients = control_df['client_id'].nunique()
    
    # Clientes que completaron el proceso (llegaron a 'confirm')
    test_completions = test_df[test_df['process_step'] == 'confirm']['client_id'].nunique()
    control_completions = control_df[control_df['process_step'] == 'confirm']['client_id'].nunique()

    # Seguridad: no permitir divisiones por cero
    if unique_test_clients == 0 or unique_control_clients == 0:
        raise ValueError("Uno de los grupos no contiene clientes únicos.")
    
    # Prueba z de proporciones
    counts = np.array([test_completions, control_completions])
    nobs = np.array([unique_test_clients, unique_control_clients])
    
    stat, p_value = proportions_ztest(counts, nobs, alternative='larger')

    # Decisión
    if p_value < alpha:
        decision = "Rechaza la hipótesis nula: hay una diferencia significativa en las tasas de finalización."
    else:
        decision = "No se rechaza la hipótesis nula: no hay una diferencia significativa en las tasas de finalización."

    return {
        'z_statistic': stat,
        'p_value': p_value,
        'decision': decision,
        'test_completion_rate': test_completions / unique_test_clients,
        'control_completion_rate': control_completions / unique_control_clients
    }
    

def hypothesis_testing_finalizacion_threshold (df, cost_effectiveness_threshold=0.05, alpha=0.05):
    """ Realiza una prueba de hipótesis para comparar las tasas de finalización entre los grupos Test y Control,
    ajustando la tasa de finalización del grupo Control con un umbral de rentabilidad.
    Parámetros:
        df (DataFrame): DataFrame que contiene los datos de clientes y sus pasos de proceso.
        cost_effectiveness_threshold (float): Umbral de rentabilidad para ajustar la tasa de finalización del grupo Control.
        alpha (float): Nivel de significancia para la prueba (por defecto 0.05).
    Retorna:
        dict: Un diccionario con el estadístico z, el valor p, la decisión de la prueba y las tasas de finalización.
    """

    # Filtrar los grupos
    df_test = df[df['Variation'] == 'Test']
    df_control = df[df['Variation'] == 'Control']

    # Clientes únicos en cada grupo
    test_total = df_test['client_id'].nunique()
    control_total = df_control['client_id'].nunique()

    # Completaciones: clientes que llegaron al paso 'confirm'
    test_completions = df_test[df_test['process_step'] == 'confirm']['client_id'].nunique()
    control_completions = df_control[df_control['process_step'] == 'confirm']['client_id'].nunique()

    # Calcular tasa de completación del grupo control y ajustarla con el umbral
    control_rate = control_completions / control_total if control_total > 0 else 0
    adjusted_control_rate = control_rate + cost_effectiveness_threshold

    # Prueba de proporciones con tasa esperada ajustada
    stat, p_value = proportions_ztest(
        [test_completions, control_completions],
        [test_total, control_total],
        value=adjusted_control_rate,
        alternative='larger')

    # Decisión
    if p_value < alpha:
        hypothesis_decision = (
            f"Rechaza la hipótesis nula: el aumento en la tasa de completación es significativo y "
            f"supera el umbral establecido por la empresa."
        )
    else:
        hypothesis_decision = (
            f"No se rechaza la hipótesis nula: el aumento no supera el umbral establecido por la empresa."
        )

    return {
        'z_statistic': round(stat, 4),
        'p_value': round(p_value, 4),
        'test_completion_rate': round(test_completions / test_total * 100, 2),
        'control_completion_rate': round(control_rate * 100, 2),
        'adjusted_control_rate': round(adjusted_control_rate * 100, 2),
        'decision': hypothesis_decision
    }

def hypothesis_test_pasos(df, alpha=0.05):
    
    """ Realiza una prueba t para comparar los tiempos de cada paso del proceso entre los grupos Control y Test
    Parámetros:
        df (DataFrame): DataFrame que contiene los datos de clientes y sus pasos de proceso.
        alpha (float): Nivel de significancia para la prueba (por defecto 0.05).
    Retorna:
        DataFrame: Un DataFrame con los resultados de la prueba t para cada paso del proceso.
    """
    

    # Lista para almacenar los resultados de cada paso
    results = []
    
    # Obtener todos los pasos únicos del proceso
    process_steps = df['process_step'].unique()
    
    # Iterar sobre cada paso
    for step in process_steps:
        # Separar los datos por grupo y paso del proceso
        control_times = df[(df['Variation'] == 'Control') & (df['process_step'] == step)]['time_diff_step']
        test_times = df[(df['Variation'] == 'Test') & (df['process_step'] == step)]['time_diff_step']
        
        # Prueba t para dos muestras independientes
        t_stat, p_value = stats.ttest_ind(control_times, test_times, equal_var=False, nan_policy='omit')
        
        # Decisión sobre la hipótesis nula
        reject_null = p_value < alpha
        decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"
        
        # Agregar los resultados a la lista
        results.append({
            'process_step': step,
            'Control Mean Time': round(control_times.mean(), 2),
            'Test Mean Time': round(test_times.mean(), 2),
            'T-Statistic': round(t_stat, 4),
            'P-Value': round(p_value, 4),
            'Alpha': alpha,
            'result': decision
        })
    
    # Convertir la lista de resultados a un DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def hypothesis_test_tasa_de_error (df, alpha=0.05):
    """ Realiza una prueba de hipótesis para comparar las tasas de error entre los grupos Control y Test.
    Parámetros:
        df (DataFrame): DataFrame que contiene los datos de clientes y sus errores.
        alpha (float): Nivel de significancia para la prueba (por defecto 0.05).
    Retorna:
        DataFrame: Un DataFrame con los resultados de la prueba de hipótesis.
    """
    
    # Filtrar por grupo Control y calcular métricas
    control_df = df[df['Variation'] == 'Control']
    n_control = len(control_df)
    errors_control = control_df['is_error'].sum()
    p_control = errors_control / n_control if n_control > 0 else 0

    # Filtrar por grupo Test y calcular métricas
    test_df = df[df['Variation'] == 'Test']
    n_test = len(test_df)
    errors_test = test_df['is_error'].sum()
    p_test = errors_test / n_test if n_test > 0 else 0

    # Validar que ambos grupos tienen datos
    if n_control == 0 or n_test == 0:
        raise ValueError("Uno de los grupos (Control o Test) no contiene registros.")

    # Proporción combinada y error estándar
    p_combined = (errors_control + errors_test) / (n_control + n_test)
    se_combined = np.sqrt(p_combined * (1 - p_combined) * (1/n_control + 1/n_test))

    # Estadístico Z y p-valor (bilateral)
    z_stat = (p_control - p_test) / se_combined
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Decisión sobre la hipótesis nula
    reject_null = p_value < alpha
    decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"
    explanation = (
        "Las diferencias en las tasas de error son estadísticamente significativas."
        if reject_null else
        "No hay evidencia suficiente para decir que las tasas de error son significativamente diferentes."
    )

    # Armar resultados en un diccionario y convertir a DataFrame si se desea
    results = {
        'Control Error Rate (%)': round(p_control * 100, 2),
        'Test Error Rate (%)': round(p_test * 100, 2),
        'Z-Statistic': round(z_stat, 4),
        'P-Value': round(p_value, 4),
        'Alpha': alpha,
        'Decision': decision,
        'Explanation': explanation
    }

    return pd.DataFrame([results])

def hypothesis_test_tasa_error_edad(df, alpha=0.05):
    """
    Realiza una prueba de hipótesis para comparar las tasas de error entre los segmentos de edad.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y segmentos de edad.
                        Debe tener las columnas 'age_segment', 'Variation' ('Control' o 'Test') y 'is_error' (0 o 1).
        alpha (float): Nivel de significancia para la prueba.
        
    Returns:
        DataFrame: Resultados de la prueba de hipótesis por segmento de edad.
    """
    age_segments = df['age_segment'].unique()
    results = []

    for segment in age_segments:
        segment_df = df[df['age_segment'] == segment]
        
        control_df = segment_df[segment_df['Variation'] == 'Control']
        test_df = segment_df[segment_df['Variation'] == 'Test']
        
        n_control = len(control_df)
        errors_control = control_df['is_error'].sum()
        p_control = errors_control / n_control if n_control > 0 else 0
        
        n_test = len(test_df)
        errors_test = test_df['is_error'].sum()
        p_test = errors_test / n_test if n_test > 0 else 0

        # Validar que ambos grupos tienen datos para comparar
        if n_control == 0 or n_test == 0:
            continue

        # Proporción combinada y error estándar para prueba de diferencia de proporciones
        p_combined = (errors_control + errors_test) / (n_control + n_test)
        se_combined = np.sqrt(p_combined * (1 - p_combined) * (1/n_control + 1/n_test))

        # Estadístico Z y p-valor (bilateral)
        z_stat = (p_control - p_test) / se_combined
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        reject_null = p_value < alpha
        decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"
        explanation = (
            "Las diferencias en las tasas de error son estadísticamente significativas"
            if reject_null else
            "No hay evidencia suficiente para decir que las tasas de error son significativamente diferentes"
        )
        
        results.append({
            'Age Segment': segment,
            'Control Error Rate (%)': round(p_control * 100, 2),
            'Test Error Rate (%)': round(p_test * 100, 2),
            'Z-Statistic': round(z_stat, 4),
            'P-Value': round(p_value, 4),
            'Alpha': alpha,
            'Decision': decision,
            'Explanation': explanation
        })

    return pd.DataFrame(results)

def hypothesis_test_tasa_error_permanencia(df, alpha=0.05):
    """
    Realiza una prueba de hipótesis para comparar las tasas de error entre los segmentos de tenencia.
    
    Parameters:
        df (DataFrame): DataFrame que contiene la información sobre errores y segmentos de tenencia.
                        Debe tener las columnas 'tenure_segment', 'Variation' ('Control' o 'Test') y 'is_error' (0 o 1).
        alpha (float): Nivel de significancia para la prueba.
        
    Returns:
        DataFrame: Resultados de la prueba de hipótesis por segmento de tenencia.
    """
    tenure_segments = df['tenure_segment'].unique()
    results = []

    for segment in tenure_segments:
        segment_df = df[df['tenure_segment'] == segment]
        
        control_df = segment_df[segment_df['Variation'] == 'Control']
        test_df = segment_df[segment_df['Variation'] == 'Test']
        
        n_control = len(control_df)
        errors_control = control_df['is_error'].sum()
        p_control = errors_control / n_control if n_control > 0 else 0
        
        n_test = len(test_df)
        errors_test = test_df['is_error'].sum()
        p_test = errors_test / n_test if n_test > 0 else 0

        # Validar que ambos grupos tienen datos para comparar
        if n_control == 0 or n_test == 0:
            continue

        # Proporción combinada y error estándar para prueba de diferencia de proporciones
        p_combined = (errors_control + errors_test) / (n_control + n_test)
        se_combined = np.sqrt(p_combined * (1 - p_combined) * (1/n_control + 1/n_test))

        # Estadístico Z y p-valor (bilateral)
        z_stat = (p_control - p_test) / se_combined
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        reject_null = p_value < alpha
        decision = "Se rechaza H₀" if reject_null else "No se rechaza H₀"
        explanation = (
            "Las diferencias en las tasas de error son estadísticamente significativas"
            if reject_null else
            "No hay evidencia suficiente para decir que las tasas de error son significativamente diferentes"
        )
        results.append({
            'Tenure Segment': segment,
            'Control Error Rate (%)': round(p_control * 100, 2),
            'Test Error Rate (%)': round(p_test * 100, 2),
            'Z-Statistic': round(z_stat, 4),
            'P-Value': round(p_value, 4),
            'Alpha': alpha,
            'Decision': decision,
            'Explanation': explanation
        })
    return pd.DataFrame(results)

# --------------------------------------------------------- #
# Funciones de exportación de datos y tablas para Tableau
# --------------------------------------------------------- #

def crear_tablas_dim_fact(df):
    """
    Crea las tablas de dimensiones y hechos para modelo estrella a partir del dataframe principal.

    Parámetros:
        df (DataFrame): DataFrame principal con todas las columnas necesarias.

    Retorna:
        tuple: (dim_client, dim_step, dim_variation, fact_process_steps)
    """
    # ----------------------- #
    # DIM_CLIENT
    # ----------------------- #
    dim_client = df[[
        'client_id', 'clnt_age', 'clnt_tenure_yr', 'gendr',
        'num_accts', 'bal', 'calls_6_mnth', 'logons_6_mnth',
        'tenure_segment', 'age_segment', 'bal_category'
    ]].drop_duplicates()

    # -----------------------#
    # DIM_STEP
    # -----------------------#
    step_order_map = {
        'start': 1, 'step_1': 2, 'step_2': 3, 'step_3': 4,
        'confirm': 5
    }
    dim_step = (
        df[['process_step']]
        .dropna()
        .drop_duplicates()
        .assign(step_order=lambda d: d['process_step'].map(step_order_map),
                is_final=lambda d: d['process_step'].isin(['confirm']))
    )

    # -----------------------#
    # DIM_VARIATION
    # -----------------------#
    dim_variation = df[['Variation']].dropna().drop_duplicates()
    dim_variation = dim_variation.rename(columns={'Variation': 'variation'})
    dim_variation['description'] = [
        'Grupo de prueba' if v == 'Test' else 'Grupo de control'
        for v in dim_variation['variation']
    ]

    # -----------------------#
    # FACT_PROCESS_STEPS
    # -----------------------#
    fact_process_steps = df[[
        'visit_id', 'visitor_id', 'client_id', 'process_step', 'date_time',
        'is_complete', 'time_diff_step', 'time_before_step', 'is_error', 'Variation'
    ]].copy()
    fact_process_steps['step_order'] = fact_process_steps['process_step'].map(step_order_map)

    return dim_client, dim_step, dim_variation, fact_process_steps