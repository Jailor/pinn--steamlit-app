import streamlit as st


def go_back(current_tab: str):
    if current_tab == "uploaded_file":
        st.session_state.pop('uploaded_file')
    elif current_tab == "partial_columns_selected":
        st.session_state.pop('filtered_df_initial')
        st.session_state.pop('timestamp_column')
        st.session_state.pop('heating_load_column')
    elif current_tab == "full_columns_selected":
        st.session_state.pop('filtered_df')
        st.session_state.pop('processed_df')
        st.session_state.pop('columns')
        st.session_state.pop('columns_selected')
        if 'best_individual' in st.session_state:
            st.session_state.pop('best_individual')
        if 'best_individual_so_far' in st.session_state:
            st.session_state.pop('best_individual_so_far')
    elif current_tab == "models_trained_uploaded":
        st.session_state.pop('model')
        st.session_state.pop('pinn_model')
        if 'best_individual' in st.session_state:
            st.session_state.pop('best_individual')
        if 'best_individual_so_far' in st.session_state:
            st.session_state.pop('best_individual_so_far')

    st.rerun()

def reset_state_and_prompt():
    st.session_state.pop('uploaded_file', None)
    st.session_state.pop('initial_df', None)
    st.session_state['error_message'] = "This dataset is not valid. Please upload a valid CSV file."
    st.rerun()