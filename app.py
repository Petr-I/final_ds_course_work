import streamlit as st


def main():
    st.set_page_config(page_title="Predictive Maintenance", layout="wide")

    pages = {
        "Анализ и модель": "analysis_and_model.py",
        "Презентация": "presentation.py"
    }

    st.sidebar.title("Навигация")
    selected_page = st.sidebar.radio("Выберите страницу", pages.keys())

    if selected_page == "Анализ и модель":
        from analysis_and_model import analysis_page
        analysis_page()
    else:
        from presentation import presentation_page
        presentation_page()


if __name__ == "__main__":
    main()