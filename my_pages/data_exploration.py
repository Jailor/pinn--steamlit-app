import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def show_data_exploration_statistics(initial_df, feature_importance_df):
    st.title('Data Exploration')

    # Add the information section
    st.subheader('Data Description and Insights')
    st.markdown("""
           **Hot Water Temp (F):**
           Hot Water Temp (F) is the temperature of the water after it has been heated by the heat pump. It indicates how hot the water is when it leaves the heat pump or when it's ready to be used or stored.

           **Cold Water Temp (F):**
           Cold Water Temp (F) represents the temperature of the outside cold-water in Fahrenheit..

           **Inlet Temp (F):**
           Inlet Temp (F) represents the temperature of the water at the inlet in Fahrenheit. It is the temperature of the water before it is heated by the heat pump, useful for calculating the heat output of the system.

           **Water Flow (Gallons):**
           The flow of water in gallons. A higher flow rate means more water is being heated, which affects the heating output of the system.

           **Watts:**
           Watts  represents power usage of the HPWH in watts. This measurement can be used to understand how much electrical energy the system uses to operate.

           **Heat Trace (W):**
           Heat tracing is a method used to maintain or raise the temperature of pipes and vessels using specially designed heating cables or tapes. The purpose of heat tracing is to prevent freezing, maintain fluidity by keeping the temperature above a certain level, or protect sensitive equipment from cold temperatures. The term "Heat Trace" in watts would then refer to the power consumption of the heat tracing system.

           **Water Heating Load (Btu):**
           System power of the HVAC unit.
           """)

    st.subheader('Dataset Summary')
    st.write(initial_df.describe())

    st.subheader('Correlation Matrix')
    corr_matrix = initial_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='hot',
        aspect='auto'
    )
    fig_corr.update_layout(
        xaxis_title='Features',
        yaxis_title='Features',
        title='Correlation Matrix'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader('Feature Importance')
    # feature importance previously calculated
    fig_importance = px.bar(feature_importance_df, x='Feature', y='Importance')
    fig_importance.update_yaxes(type='log')
    st.plotly_chart(fig_importance)

    # Add histograms of the data
    st.subheader('Data Distributions')

    # Loop through each column in the dataframe and plot histograms
    for col in initial_df.columns:
        fig_hist = px.histogram(initial_df, x=col, nbins=30, title=f'Distribution of {col}')
        fig_hist.update_layout(
            xaxis_title=col,
            yaxis_title='Count',
            title=f'Distribution of {col}'
        )
        st.plotly_chart(fig_hist)

    # Add scatter matrix plot

    short_column_names = {
        'Hot Water Temp (F)': 'Hot Temp',
        'Cold Water Temp (F)': 'Cold Temp',
        'Inlet Temp (F)': 'Inlet Temp',
        'Water Flow (Gallons)': 'Water Flow',
        'Watts': 'Watts',
        'Heat Trace (W)': 'Heat Trace',
        'Water Heating Load (Btu)': 'Heating Load'
    }
    plot_df_short = initial_df.rename(columns=short_column_names)
    st.subheader('Scatter Matrix Plot')
    fig_scatter_matrix = px.scatter_matrix(plot_df_short)
    fig_scatter_matrix.update_layout(
        title='Scatter Matrix Plot',
        width=1000,
        height=1000,
    )
    st.plotly_chart(fig_scatter_matrix)

    # Add heatmap for missing values
    st.subheader('Heatmap of Missing Values')
    fig_missing = px.imshow(initial_df.isna(), color_continuous_scale='viridis', aspect='auto')
    fig_missing.update_layout(
        xaxis_title='Features',
        yaxis_title='Samples',
        title='Heatmap of Missing Values'
    )
    st.plotly_chart(fig_missing)


def show_data_exploration_time_series_analysis_generic(initial_df):
    columns = initial_df.columns
    with st.expander('Time Series Analysis (Weekly Aggregated)', expanded=True):
        initial_df_resampled = initial_df.resample('W').mean()
        for col in columns:
            fig_ts = px.line(initial_df_resampled, x=initial_df_resampled.index, y=col,
                             title=f'Weekly Time Series of {col}')
            fig_ts.update_layout(
                xaxis_title='Time',
                yaxis_title=col,
                title=f'Weekly Time Series of {col}',
                yaxis=dict(tickformat=".2f")
            )
            st.plotly_chart(fig_ts)

    with st.expander('Time Series Analysis (Interactive)', expanded=False):
        for col in columns:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=initial_df.index, y=initial_df[col], mode='lines', name=col))
            fig_ts.update_layout(
                title=f'Time Series of {col}',
                xaxis_title='Time',
                yaxis_title=col,
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    ),
                    type="date"
                )
            )
            st.plotly_chart(fig_ts)


    with st.expander('Time Series Analysis (Smoothed with Moving Average)', expanded=False):
        # Add a slider for the moving average window size
        window_size = st.slider('Select Moving Average Window Size', min_value=1, max_value=60, value=21, step=1)

        initial_df_ma = initial_df.copy()
        for col in columns:
            initial_df_ma[f'{col}_MA'] = initial_df_ma[col].rolling(window=window_size).mean()
            fig_ts = px.line(initial_df_ma, x=initial_df_ma.index, y=f'{col}_MA',
                             title=f'Moving Average Time Series of {col}')
            fig_ts.update_layout(
                xaxis_title='Time',
                yaxis_title=col,
                title=f'Moving Average Time Series of {col} (Window Size: {window_size})',
                yaxis=dict(tickformat=".2f")
            )
            st.plotly_chart(fig_ts)


def show_data_exploration_statistics_generic(initial_df, feature_importance_df=None):
    st.title('Data Exploration')

    st.subheader('Dataset Summary')
    st.write(initial_df.describe())

    st.subheader('Correlation Matrix')
    corr_matrix = initial_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='hot',
        aspect='auto'
    )
    fig_corr.update_layout(
        xaxis_title='Features',
        yaxis_title='Features',
        title='Correlation Matrix'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    if feature_importance_df is not None:
        st.subheader('Feature Importance')
        # feature importance previously calculated
        fig_importance = px.bar(feature_importance_df, x='Feature', y='Importance')
        fig_importance.update_yaxes(type='log')
        st.plotly_chart(fig_importance)

    # Add histograms of the data
    st.subheader('Data Distributions')

    # Loop through each column in the dataframe and plot histograms
    for col in initial_df.columns:
        fig_hist = px.histogram(initial_df, x=col, nbins=30, title=f'Distribution of {col}')
        fig_hist.update_layout(
            xaxis_title=col,
            yaxis_title='Count',
            title=f'Distribution of {col}'
        )
        st.plotly_chart(fig_hist)

    st.subheader('Scatter Matrix Plot')
    fig_scatter_matrix = px.scatter_matrix(initial_df)
    fig_scatter_matrix.update_layout(
        title='Scatter Matrix Plot',
        width=1000,
        height=1000,
    )
    st.plotly_chart(fig_scatter_matrix)

    # Add heatmap for missing values
    st.subheader('Heatmap of Missing Values')
    fig_missing = px.imshow(initial_df.isna(), color_continuous_scale='viridis', aspect='auto')
    fig_missing.update_layout(
        xaxis_title='Features',
        yaxis_title='Samples',
        title='Heatmap of Missing Values'
    )
    st.plotly_chart(fig_missing)
