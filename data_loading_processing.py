import difflib

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


def load_existing_datasets():
    if 'datasets' in st.session_state:
        return
    rossland_original = load_dataset('rossland_original')
    # rossland_renamed_df = load_dataset('rossland_renamed')
    # cleaned_v1_e350 = load_dataset('cleaned_v1_e350')
    st.session_state['rossland_df'] = rossland_original
    st.session_state['datasets'] = {
        'Rossland Site 1 E350': rossland_original,
        # 'Rossland Site 1 renamed E350': rossland_renamed_df,
        # 'Victoria Island 1 E350': cleaned_v1_e350
    }


def load_dataset(dataset_name):
    try:
        df = pd.read_csv(f'datasets/{dataset_name}.csv', low_memory=False)
        return df
    except Exception as e:
        print(f"An error occurred while reading Rossland: {e}")
        return None


@st.cache_resource
def load_data(uploaded_file):
    df = None
    try:
        # Check if the file is not empty
        if uploaded_file.size == 0:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            return None

        # Try reading the file with the specified delimiter
        df = pd.read_csv(uploaded_file, low_memory=False)
        if df.empty:
            print("No data found in the file. Please check the file content.")
            return None
    except pd.errors.EmptyDataError:
        print("The uploaded file has no columns to parse. Please check the delimiter and file format.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    return df


def process_existing_dataset(df, dataset_name):
    if dataset_name == 'Rossland Site 1 E350':
        # Directly process the Rossland site 1 data according to predefined rules
        # Drop bad data
        df.drop('ID', axis=1, inplace=True)
        df.drop('Uncorrected Water Flow (Gallons)', axis=1, inplace=True)
        df.drop('Uncorrected Hot Water Temp (F)', axis=1, inplace=True)
        df.drop('Uncorrected Cold Water Temp (F)', axis=1, inplace=True)
        df.drop('Site', axis=1, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        df['Watts'] = df['Watts'].astype(str)
        df['Water Heating Load (Btu)'] = df['Water Heating Load (Btu)'].astype(str)

        # Convert non-numeric values to NaN
        df['Watts'] = pd.to_numeric(df['Watts'], errors='coerce')
        df['Water Heating Load (Btu)'] = pd.to_numeric(df['Water Heating Load (Btu)'], errors='coerce')
        df.dropna(inplace=True)

        # Resample the data to hourly intervals
        df = df.resample('h').agg({
            'Hot Water Temp (F)': 'mean',
            'Cold Water Temp (F)': 'mean',
            'Water Flow (Gallons)': 'sum',
            'Inlet Temp (F)': 'mean',
            'Watts': 'mean',
            'Heat Trace (W)': 'mean',
            'Water Heating Load (Btu)': 'sum'
        })
        df.dropna(inplace=True)
        return df
    else:
        return df


@st.cache_data
def get_feature_importances(initial_df, heating_load_column='Water Heating Load (Btu)'):
    X = initial_df.drop(columns=[heating_load_column])
    y = initial_df[heating_load_column]
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    return feature_importance_df


def get_most_similar_column(target, columns):
    matches = difflib.get_close_matches(target, columns, n=1, cutoff=0.0)
    return matches[0] if matches else None


def prompt_user_for_columns(df):
    st.title('Process Dataset')
    st.subheader("Dataset Columns")
    st.markdown("""
            Please select the columns to keep and map the necessary columns for the analysis. The kept columns must necessarily match 
             at least the following columns in order to apply physics-informed machine learning for heating load:
              - **water flow** => The amount of water entering the HPWH system
              - **outlet water temp** => The temperature of the water leaving the HPWH system
              - **inlet temp** => The temperature of the water entering the HPWH system
              - **timestamp** => The timestamp of the data. Should be in format YYYY-MM-DD HH:MM:SS
              - **heating load** => The heating load of the HPWH system \n
            After this step, you will be able to upload/train models and view performance metrics.
              """)

    columns_to_keep = st.multiselect("Select columns to keep (Minimum 5)", options=df.columns,
                                     default=df.columns.tolist())

    if len(columns_to_keep) < 5:
        st.error("Please select at least 5 columns to proceed.")
        return

    df = df[columns_to_keep]

    # Find the most similar column names
    default_water_flow_column = get_most_similar_column("water flow", columns_to_keep)
    default_outlet_water_temp_column = get_most_similar_column("outlet water temp", columns_to_keep)
    default_inlet_temp_column = get_most_similar_column("inlet temp", columns_to_keep)
    default_timestamp_column = get_most_similar_column("timestamp", columns_to_keep)
    default_heating_load_column = get_most_similar_column("heating load", columns_to_keep)

    submitted = False
    try:
        with st.form("column_mapping"):
            water_flow_column = st.selectbox("Select the column for water flow", options=columns_to_keep,
                                             index=columns_to_keep.index(
                                                 default_water_flow_column) if default_water_flow_column else 0)
            outlet_water_temp_column = st.selectbox("Select the column for outlet water temperature",
                                                    options=columns_to_keep, index=columns_to_keep.index(
                    default_outlet_water_temp_column) if default_outlet_water_temp_column else 0)
            inlet_temp_column = st.selectbox("Select the column for inlet temperature", options=columns_to_keep,
                                             index=columns_to_keep.index(
                                                 default_inlet_temp_column) if default_inlet_temp_column else 0)
            timestamp_column = st.selectbox("Select the column for timestamp", options=columns_to_keep,
                                            index=columns_to_keep.index(
                                                default_timestamp_column) if default_timestamp_column else 0)
            heating_load_column = st.selectbox("Select the column for heating load", options=columns_to_keep,
                                               index=columns_to_keep.index(
                                                   default_heating_load_column) if default_heating_load_column else 0)
            submitted = st.form_submit_button("Submit")

        if submitted:
            st.session_state['columns_selected'] = True
            st.session_state['columns'] = {
                'water_flow': water_flow_column,
                'outlet_water_temp': outlet_water_temp_column,
                'inlet_temp': inlet_temp_column,
                'timestamp': timestamp_column,
                'heating_load': heating_load_column
            }
            st.session_state['filtered_df'] = df
            st.rerun()
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")


@st.cache_data
def process_new_dataset(df, water_flow_column, outlet_water_temp_column, inlet_temp_column, timestamp_column,
                        heating_load_column):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df.set_index(timestamp_column, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop columns that are entirely NaN
    df.dropna(axis=1, how='all', inplace=True)

    # Create a dictionary for aggregation
    agg_dict = {
        water_flow_column: 'sum',
        outlet_water_temp_column: 'mean',
        inlet_temp_column: 'mean',
        heating_load_column: 'sum'
    }

    # Add other columns to be resampled by their mean
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = 'mean'

    # Resample the data to hourly intervals
    df = df.resample('h').agg(agg_dict)
    df.dropna(inplace=True)
    return df


def prompt_user_for_partial_columns(df):
    st.title('Map Timestamp and Heating Load Columns')
    st.write("The form below requires you to choose the columns that represent the timestamp of each entry in your "
             "dataset (must necessarily be a date-time column of format YYYY-MM-DD HH:MM:SS and the column that "
             "represents the heating load of the HPWH system. The application will then transition and offer dataset"
             "processing and statistics.")

    # Find the most similar column names
    columns = df.columns.tolist()

    default_timestamp_column = get_most_similar_column("timestamp", columns)
    default_heating_load_column = get_most_similar_column("heating load", columns)

    submitted = False
    try:
        with st.form("column_mapping_initial"):

            timestamp_column = st.selectbox("Select the column for timestamp", options=columns,
                                            index=columns.index(
                                                default_timestamp_column) if default_timestamp_column else 0)
            heating_load_column = st.selectbox("Select the column for heating load", options=columns,
                                               index=columns.index(
                                                   default_heating_load_column) if default_heating_load_column else 0)
            submitted = st.form_submit_button("Submit")

        if submitted:
            df_copy = df.copy()

            df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
            df_copy.set_index(timestamp_column, inplace=True)

            for col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

            # Drop columns that are entirely NaN
            df_copy.dropna(axis=1, how='all', inplace=True)

            # Reduce df_copy to a maximum of 50000 entries
            if len(df_copy) > 50000:
                df_copy = df_copy.iloc[:50000]
                st.session_state['is_reduced_dataset'] = True
            else:
                st.session_state['is_reduced_dataset'] = False

            st.session_state['filtered_df_initial'] = df_copy
            st.session_state['timestamp_column'] = timestamp_column
            st.session_state['heating_load_column'] = heating_load_column
            st.rerun()
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")