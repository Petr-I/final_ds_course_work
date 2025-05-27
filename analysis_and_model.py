import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def analysis_page():
    st.title("📊 Анализ данных и модель")

    # Загрузка данных
    with st.expander("Загрузить данные"):
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)

            # Предобработка
            data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

            numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                              'Torque [Nm]', 'Tool wear [min]']

            scaler = StandardScaler()
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

            X = data.drop(columns=['Machine failure'])
            y = data['Machine failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            st.subheader("Результаты оценки модели")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**ROC-AUC:** {auc:.2f}")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Ввод данных для прогноза")
            with st.form("input_form"):
                input_type = st.selectbox("Тип (Type)", options=['L', 'M', 'H'])
                air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                process_temp = st.number_input("Температура процесса [K]", value=310.0)
                rotation_speed = st.number_input("Скорость вращения [rpm]", value=1500.0)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]", value=100.0)
                submitted = st.form_submit_button("Сделать прогноз")

            if submitted:
                # Преобразование данных в DataFrame
                input_dict = {
                    'Type': [input_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotation_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear],
                }

                input_df = pd.DataFrame(input_dict)
                input_df['Type'] = input_df['Type'].map({'L': 0, 'M': 1, 'H': 2})

                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                st.markdown("### Результат предсказания:")
                st.write("**Отказ оборудования**" if prediction == 1 else "**Оборудование работает нормально**")
                st.write(f"**Вероятность отказа:** {probability:.2f}")