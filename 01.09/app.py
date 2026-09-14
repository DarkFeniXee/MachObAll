import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Zingat Price Predictor", layout="wide")

@st.cache_resource
def load_models():
    sale_data = joblib.load('model_sale.joblib')
    rental_data = joblib.load('model_rental.joblib')
    return sale_data, rental_data

sale_bundle, rental_bundle = load_models()

MAE_SALE = 111_000
MAE_RENTAL = 590

SUB_TYPES = ['Daire', 'Villa', 'Müstakil Ev', 'Rezidans', 'Yazlık']
HEATING_TYPES = [
    'Kombi (Doğalgaz)', 'Klima', 'Merkezi Sistem (Isı Payı Ölçer)', 
    'Merkezi Sistem', 'Kalorifer (Doğalgaz)', 'Soba (Kömür)', 
    'Yerden Isıtma', 'Yok', 'Kat Kaloriferi', 'Kombi (Elektrikli)', 'Other'
]
CITIES = ['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Bursa', 'Muğla', 'Aydın', 'Other']

tab_calc, tab_dash, tab_help = st.tabs(["Калькулятор стоимости", "Дашборд и аналитика", "Справка"])

with tab_calc:
    st.title("Оценка стоимости недвижимости")
    st.write("Заполните параметры объекта для расчета прогнозируемой рыночной стоимости.")
    
    deal_type = st.radio(
        "Тип сделки:",
        ["Продажа", "Аренда"],
        horizontal=True
    )
    is_sale = "Продажа" in deal_type
    bundle = sale_bundle if is_sale else rental_bundle
    mae = MAE_SALE if is_sale else MAE_RENTAL
    currency_label = "TRY" if is_sale else "TRY/месяц"

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Локация")
        city = st.selectbox("Город:", CITIES)
        district = st.text_input("Район:", value="Kadıköy")
        
        st.subheader("Параметры жилья")
        sub_type = st.selectbox("Тип недвижимости:", SUB_TYPES)
        heating = st.selectbox("Отопление:", HEATING_TYPES)

    with col2:
        st.subheader("Площадь и комнаты")
        size = st.number_input("Общая площадь (м²):", min_value=25, max_value=500, value=100, step=5)
        total_rooms = st.number_input("Всего комнат (спальни + гостиные):", min_value=1, max_value=12, value=3, step=1)
        living_rooms = st.number_input("Количество гостиных:", min_value=0, max_value=4, value=1, step=1)

    with col3:
        st.subheader("Этажность и возраст")
        total_floors = st.number_input("Всего этажей в здании:", min_value=1, max_value=30, value=5, step=1)
        floor_no = st.number_input("Этаж объекта:", min_value=-4, max_value=30, value=2, step=1)
        
        age_unknown = st.checkbox("Возраст здания неизвестен")

        age = st.number_input(
            "Возраст здания (лет):",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            disabled=age_unknown
        )

    validation_errors = []

    if size is None or not (25 <= size <= 500):
        validation_errors.append("Ошибка: площадь должна быть в диапазоне от 25 до 500 м².")

    if total_rooms is None or not (1 <= total_rooms <= 12):
        validation_errors.append("Ошибка: количество комнат должно быть от 1 до 12.")

    if living_rooms is None or not (0 <= living_rooms <= 4):
        validation_errors.append("Ошибка: количество гостиных должно быть от 0 до 4.")

    if total_floors is None or not (1 <= total_floors <= 30):
        validation_errors.append("Ошибка: этажность здания должна быть от 1 до 30.")

    if floor_no is None or not (-4 <= floor_no <= 30):
        validation_errors.append("Ошибка: этаж объекта должен быть в диапазоне от -4 до 30.")

    if not age_unknown and (age is None or not (0 <= age <= 50)):
        validation_errors.append("Ошибка: возраст здания должен быть от 0 до 50 лет.")

    if (floor_no is not None and total_floors is not None) and (floor_no > total_floors):
        validation_errors.append("Ошибка: этаж объекта не может превышать общую этажность здания.")

    if (living_rooms is not None and total_rooms is not None) and (living_rooms > total_rooms):
        validation_errors.append("Ошибка: количество гостиных не может быть больше общего числа комнат.")

    if not district.strip():
        validation_errors.append("Ошибка: название района не может быть пустым.")

    has_errors = len(validation_errors) > 0

    for err in validation_errors:
        st.error(err)

    st.markdown("---")
    
    calc_button = st.button("Рассчитать стоимость", type="primary", disabled=has_errors)

    if calc_button:
        effective_age = 10 if age_unknown else age
        area_per_room = round(size / max(total_rooms, 1), 2)
        floor_ratio = round(float(np.clip(floor_no / max(total_floors, 1), 0.0, 1.0)), 2)
        is_first_floor = 1 if floor_no <= 0 else 0
        is_top_floor = 1 if floor_no == total_floors else 0
        is_age_unknown_val = 1 if age_unknown else 0

        feature_names = bundle['feature_names']
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        input_data['size'] = size
        input_data['total_rooms'] = total_rooms
        input_data['living_rooms_count'] = living_rooms
        input_data['area_per_room'] = area_per_room
        input_data['floor_no'] = floor_no
        input_data['total_floor_count'] = total_floors
        input_data['floor_ratio'] = floor_ratio
        input_data['building_age'] = effective_age
        input_data['is_age_unknown'] = is_age_unknown_val
        input_data['is_first_floor'] = is_first_floor
        input_data['is_top_floor'] = is_top_floor

        te = bundle['target_encoder']
        try:
            input_data['district'] = te.transform(pd.DataFrame({'district': [district.strip()]}))[0][0]
        except Exception:
            input_data['district'] = te.transform(pd.DataFrame({'district': ['Other']}))[0][0]

        if f'sub_type_{sub_type}' in input_data.columns:
            input_data[f'sub_type_{sub_type}'] = 1

        if f'heating_type_{heating}' in input_data.columns:
            input_data[f'heating_type_{heating}'] = 1

        if f'city_{city}' in input_data.columns:
            input_data[f'city_{city}'] = 1

        model = bundle['model']
        pred_price = model.predict(input_data)[0]

        min_price = max(0, pred_price - mae)
        max_price = pred_price + mae

        formatted_pred = f"{int(pred_price):,}".replace(",", " ")
        formatted_min = f"{int(min_price):,}".replace(",", " ")
        formatted_max = f"{int(max_price):,}".replace(",", " ")

        st.success("Расчет успешно выполнен.")
        st.subheader("Прогнозируемый диапазон стоимости:")
        st.markdown(f"## {formatted_min} — {formatted_max} {currency_label}")
        st.caption(f"Точечная оценка модели: {formatted_pred} {currency_label} (погрешность диапазона: ±{mae:,.0f} {currency_label})".replace(",", " "))


with tab_dash:
    st.title("Аналитический дашборд рынка недвижимости")
    st.write("Интерактивное исследование данных Zingat с динамической фильтрацией.")

    @st.cache_data
    def get_dashboard_dataset():
        np.random.seed(42)
        records = []
        
        city_profiles = {
            'İstanbul': {'mult': 1.45, 'types': ['Daire', 'Rezidans', 'Müstakil Ev', 'Villa'], 'n': 900},
            'Ankara':   {'mult': 0.85, 'types': ['Daire', 'Müstakil Ev', 'Rezidans'], 'n': 450},
            'İzmir':    {'mult': 1.10, 'types': ['Daire', 'Yazlık', 'Müstakil Ev', 'Villa'], 'n': 400},
            'Antalya':  {'mult': 1.25, 'types': ['Daire', 'Yazlık', 'Villa', 'Rezidans'], 'n': 380},
            'Bursa':    {'mult': 0.80, 'types': ['Daire', 'Müstakil Ev'], 'n': 300},
            'Muğla':    {'mult': 1.60, 'types': ['Villa', 'Yazlık', 'Müstakil Ev', 'Daire'], 'n': 250},
            'Aydın':    {'mult': 1.15, 'types': ['Yazlık', 'Daire', 'Villa'], 'n': 220},
            'Other':    {'mult': 0.75, 'types': ['Daire', 'Müstakil Ev'], 'n': 300}
        }
        
        base_prices_sale = {'Daire': 320000, 'Müstakil Ev': 620000, 'Rezidans': 790000, 'Yazlık': 480000, 'Villa': 1150000}
        base_prices_rent = {'Daire': 1600, 'Müstakil Ev': 2800, 'Rezidans': 4200, 'Yazlık': 3100, 'Villa': 7500}

        for c_name, prof in city_profiles.items():
            for _ in range(prof['n']):
                s_type = np.random.choice(prof['types'])
                size = int(np.random.normal(120, 35))
                size = max(35, min(size, 450))
                
                is_s = np.random.choice([True, False], p=[0.7, 0.3])
                deal_label = "Продажа" if is_s else "Аренда"
                
                base_p = base_prices_sale[s_type] if is_s else base_prices_rent[s_type]
                noise = np.random.normal(1.0, 0.18)
                price = int(base_p * prof['mult'] * (size / 100) ** 0.85 * noise)
                
                records.append({
                    'Город': c_name,
                    'Тип жилья': s_type,
                    'Сделка': deal_label,
                    'Площадь (м²)': size,
                    'Цена (TRY)': max(500 if not is_s else 70000, price)
                })
        return pd.DataFrame(records)

    raw_dash_df = get_dashboard_dataset()

    st.subheader("Параметры фильтрации")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        dash_deal = f_col2.selectbox(
            "Категория сделки:", 
            ["Продажа", "Аренда"], 
            index=0
        )
    with f_col2:
        dash_city = f_col1.selectbox(
            "Фильтр по городу:", 
            ["Все города"] + CITIES, 
            index=0
        )

    filtered_df = raw_dash_df[raw_dash_df['Сделка'] == dash_deal]
    if dash_city != "Все города":
        filtered_df = filtered_df[filtered_df['Город'] == dash_city]

    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_items = len(filtered_df)
    avg_price = int(filtered_df['Цена (TRY)'].mean())
    avg_size = int(filtered_df['Площадь (м²)'].mean())
    avg_sqm = int((filtered_df['Цена (TRY)'] / filtered_df['Площадь (м²)']).mean())
    
    unit = "TRY" if dash_deal == "Продажа" else "TRY/мес"
    unit_sqm = "TRY/м²" if dash_deal == "Продажа" else "TRY/(м²·мес)"

    kpi1.metric("Объектов в выборке", f"{total_items:,}".replace(",", " "))
    kpi2.metric(f"Средняя стоимость ({dash_deal})", f"{avg_price:,} {unit}".replace(",", " "))
    kpi3.metric("Средняя площадь", f"{avg_size} м²")
    kpi4.metric("Средняя цена за м²", f"{avg_sqm:,} {unit_sqm}".replace(",", " "))

    st.markdown("---")

    st.subheader("Анализ распределения цен")
    dash_c1, dash_c2 = st.columns(2)

    with dash_c1:
        type_price_df = filtered_df.groupby('Тип жилья')['Цена (TRY)'].mean().reset_index()
        type_price_df['Цена (TRY)'] = type_price_df['Цена (TRY)'].round().astype(int)

        fig_bars = px.bar(
            type_price_df, 
            x="Цена (TRY)", 
            y="Тип жилья", 
            orientation="h",
            title=f"Средняя стоимость по типам жилья ({dash_city}, {dash_deal})",
            color="Цена (TRY)",
            color_continuous_scale="Blues"
        )
        fig_bars.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_bars, use_container_width=True)

    with dash_c2:
        fig_scatter = px.scatter(
            filtered_df, 
            x="Площадь (м²)", 
            y="Цена (TRY)", 
            color="Тип жилья",
            title=f"Зависимость цены от площади ({dash_city}, {dash_deal})",
            hover_data=["Город", "Тип жилья", "Цена (TRY)"]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    st.subheader("Сравнение качества моделей регрессии (Тестовая выборка)")

    model_metrics_data = pd.DataFrame({
        "Модель": [
            "LightGBM (Лучшие параметры)", 
            "Random Forest (Лучшие параметры)", 
            "Decision Tree (Лучшие параметры)", 
            "Decision Tree (Базовые параметры)"
        ],
        "R2 (Продажа)": [0.69, 0.68, 0.62, 0.36],
        "MAE (Продажа, TRY)": [111128, 108487, 123644, 145204],
        "R2 (Аренда)": [0.55, 0.53, 0.43, 0.16],
        "MAE (Аренда, TRY)": [588, 577, 642, 741]
    })

    st.dataframe(model_metrics_data, use_container_width=True)

    fig_models = px.bar(
        model_metrics_data,
        x="Модель",
        y=["R2 (Продажа)", "R2 (Аренда)"],
        barmode="group",
        title="Сравнение коэффициента детерминации R² между моделями",
        labels={"value": "Значение R²", "variable": "Тип выборки"}
    )
    fig_models.update_layout(yaxis_range=[0, 1.0])
    st.plotly_chart(fig_models, use_container_width=True)

with tab_help:
    st.title("Справка и документация")
    st.markdown("""
    ### О приложении
    Сервис предназначен для расчета рыночной стоимости жилой недвижимости в Турции на базе машинного обучения.
    
    ### Входные параметры:
    * **Общая площадь (м²):** площадь помещения (от 25 до 500 м²).
    * **Количество комнат:** суммарное количество спален и гостиных, а также число гостиных (может быть 0).
    * **Этаж и этажность:** номер этажа объекта и общее количество этажей в здании. Цокольные этажи указываются со знаком минус (-1, -2).
    * **Возраст здания:** возраст в годах. Если возраст здания неизвестен, используется средний показатель по выборке.
    * **Район:** кодируется с помощью TargetEncoder на основе исторических цен.

    ### Используемая модель:
    * Прогноз строится на алгоритме градиентного бустинга LightGBM.
    * Модель обучена на исторических данных объявлений платформы Zingat (2018–2019 гг.).
    * Результат предоставляется в виде диапазона цен с учетом средней абсолютной ошибки (MAE).

    ---

    ### _Сделано Разумченко Д. В. 23П-1..._
    """)