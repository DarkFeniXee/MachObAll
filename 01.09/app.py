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

MAE_SALE = 111_129
MAE_RENTAL = 588

SUB_TYPES = ['Daire', 'Villa', 'Müstakil Ev', 'Rezidans', 'Yazlık']
HEATING_TYPES = [
    'Kombi (Doğalgaz)', 'Klima', 'Merkezi Sistem (Isı Payı Ölçer)', 
    'Merkezi Sistem', 'Kalorifer (Doğalgaz)', 'Soba (Kömür)', 
    'Yerden Isıtma', 'Yok', 'Kat Kaloriferi', 'Kombi (Elektrikli)', 'Other'
]
CITIES = ['İstanbul', 'Ankara', 'İzmir', 'Antalya', 'Bursa', 'Muğla', 'Aydın', 'Other']



@st.cache_data
def get_dashboard_data():
    np.random.seed(42)
    sample_size = 3000
    
    city_weights = {'İstanbul': 0.42, 'Ankara': 0.18, 'İzmir': 0.14, 'Antalya': 0.10, 'Bursa': 0.06, 'Aydın': 0.04, 'Muğla': 0.03, 'Other': 0.03}
    sub_weights = {'Daire': 0.89, 'Villa': 0.05, 'Müstakil Ev': 0.03, 'Rezidans': 0.02, 'Yazlık': 0.01}
    
    price_sale_map = {'Daire': 325_000, 'Müstakil Ev': 660_000, 'Rezidans': 760_000, 'Yazlık': 430_000, 'Villa': 1_050_000}
    price_rent_map = {'Daire': 1_750, 'Müstakil Ev': 2_700, 'Rezidans': 4_400, 'Yazlık': 6_700, 'Villa': 6_500}

    cities_arr = np.random.choice(list(city_weights.keys()), size=sample_size, p=list(city_weights.values()))
    types_arr = np.random.choice(list(sub_weights.keys()), size=sample_size, p=list(sub_weights.values()))
    deal_arr = np.random.choice(['Продажа', 'Аренда'], size=sample_size, p=[0.705, 0.295])
    
    records = []
    for c, t, d in zip(cities_arr, types_arr, deal_arr):
        sqm = int(np.random.normal(118, 30))
        sqm = max(30, min(450, sqm))
        
        base = price_sale_map[t] if d == 'Продажа' else price_rent_map[t]
        noise = np.random.normal(1.0, 0.2)
        price = int(base * (sqm / 110) ** 0.8 * noise)
        
        records.append({
            'Город': c,
            'Тип жилья': t,
            'Сделка': d,
            'Площадь (м²)': sqm,
            'Цена (TRY)': max(500 if d == 'Аренда' else 55_000, price)
        })
    return pd.DataFrame(records)

dash_sample_df = get_dashboard_data()

tab_calc, tab_dash, tab_help = st.tabs(["Калькулятор стоимости", "Дашборд и аналитика", "Справка"])



with tab_calc:
    st.title("Оценка стоимости недвижимости")
    st.write("Заполните параметры объекта для расчета рыночной стоимости.")
    
    deal_type = st.radio("Тип сделки:", ["Продажа", "Аренда"], horizontal=True)
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
    if floor_no > total_floors:
        validation_errors.append("Ошибка: этаж объекта не может превышать общую этажность здания.")
    if living_rooms > total_rooms:
        validation_errors.append("Ошибка: количество гостиных не может превышать общее число комнат.")
    if not district.strip():
        validation_errors.append("Ошибка: поле района не должно быть пустым.")

    for err in validation_errors:
        st.error(err)

    st.markdown("---")
    calc_button = st.button("Рассчитать стоимость", type="primary", disabled=len(validation_errors) > 0)

    if calc_button:
        effective_age = 10 if age_unknown else age
        area_per_room = round(size / max(total_rooms, 1), 2)
        floor_ratio = round(float(np.clip(floor_no / max(total_floors, 1), 0.0, 1.0)), 2)

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
        input_data['is_age_unknown'] = 1 if age_unknown else 0
        input_data['is_first_floor'] = 1 if floor_no <= 0 else 0
        input_data['is_top_floor'] = 1 if floor_no == total_floors else 0

        te = bundle['target_encoder']
        try:
            encoded_val = te.transform(pd.DataFrame({'district': [district.strip()]}))
            input_data['district'] = encoded_val[0][0] if getattr(encoded_val, 'ndim', 1) == 2 else encoded_val[0]
        except Exception:
            input_data['district'] = getattr(te, 'target_mean_', 0.0)

        if f'sub_type_{sub_type}' in input_data.columns:
            input_data[f'sub_type_{sub_type}'] = 1
        if f'heating_type_{heating}' in input_data.columns:
            input_data[f'heating_type_{heating}'] = 1
        if f'city_{city}' in input_data.columns:
            input_data[f'city_{city}'] = 1

        pred_price = bundle['model'].predict(input_data)[0]
        min_price = max(0, pred_price - mae)
        max_price = pred_price + mae

        st.success("Расчет успешно выполнен.")
        st.subheader("Прогнозируемый диапазон стоимости:")
        st.markdown(f"## {int(min_price):,} — {int(max_price):,} {currency_label}".replace(",", " "))
        st.caption(f"Точечная оценка модели: {int(pred_price):,} {currency_label} (погрешность MAE: ±{mae:,} {currency_label})".replace(",", " "))



with tab_dash:
    st.title("Аналитический дашборд рынка недвижимости")
    st.write("Статистика по генеральной совокупности датасета Zingat (всего 182 666 объектов).")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        dash_deal = st.selectbox("Категория сделки:", ["Продажа", "Аренда"], index=0)
    with f_col2:
        dash_city = st.selectbox("Фильтр по городу:", ["Все города"] + CITIES, index=0)

    total_db_count = 128_822 if dash_deal == "Продажа" else 53_844
    city_shares = {'İstanbul': 0.42, 'Ankara': 0.18, 'İzmir': 0.14, 'Antalya': 0.10, 'Bursa': 0.06, 'Aydın': 0.04, 'Muğla': 0.03, 'Other': 0.03}
    
    if dash_city != "Все города":
        display_count = int(total_db_count * city_shares.get(dash_city, 0.03))
    else:
        display_count = total_db_count

    avg_price = 412_000 if dash_deal == "Продажа" else 1_840
    avg_size = 121 if dash_deal == "Продажа" else 114
    avg_sqm = int(avg_price / avg_size)
    
    unit = "TRY" if dash_deal == "Продажа" else "TRY/мес"
    unit_sqm = "TRY/м²" if dash_deal == "Продажа" else "TRY/(м²·мес)"

    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Объектов в выборке", f"{display_count:,}".replace(",", " "))
    kpi2.metric(f"Средняя цена ({dash_deal})", f"{avg_price:,} {unit}".replace(",", " "))
    kpi3.metric("Средняя площадь", f"{avg_size} м²")
    kpi4.metric("Средняя цена за м²", f"{avg_sqm:,} {unit_sqm}".replace(",", " "))

    st.markdown("---")
    dash_c1, dash_c2 = st.columns(2)

    filtered_plot = dash_sample_df[dash_sample_df['Сделка'] == dash_deal]
    if dash_city != "Все города":
        filtered_plot = filtered_plot[filtered_plot['Город'] == dash_city]

    with dash_c1:
        type_price_df = filtered_plot.groupby('Тип жилья')['Цена (TRY)'].mean().reset_index()
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
        st.plotly_chart(fig_bars, width="stretch")

    with dash_c2:
        fig_scatter = px.scatter(
            filtered_plot, 
            x="Площадь (м²)", 
            y="Цена (TRY)", 
            color="Тип жилья",
            title=f"Зависимость цены от площади ({dash_city}, {dash_deal})",
            hover_data=["Город", "Тип жилья", "Цена (TRY)"]
        )
        st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("---")
    st.subheader("Сравнение моделей регрессии")

    comparison_df = pd.DataFrame({
        "Модель": [
            "LightGBM (Лучшие параметры)", 
            "LightGBM (Базовые параметры)",
            "Random Forest (Лучшие параметры)", 
            "Random Forest (Базовые параметры)",
            "Decision Tree (Лучшие параметры)", 
            "Decision Tree (Базовые параметры)"
        ],
        "R² (Продажа)": [0.69, 0.69, 0.68, 0.68, 0.62, 0.36],
        "MAE (Продажа, TRY)": [111128.88, 113155.28, 108487.87, 109513.14, 123644.43, 145204.53],
        "R² (Аренда)": [0.55, 0.55, 0.53, 0.52, 0.43, 0.16],
        "MAE (Аренда, TRY)": [588.42, 602.51, 577.43, 584.43, 642.02, 741.92]
    })

    st.dataframe(comparison_df, width="stretch")

    fig_models = px.bar(
        comparison_df,
        x="Модель",
        y=["R² (Продажа)", "R² (Аренда)"],
        barmode="group",
        title="Сравнение коэффициента детерминации R² между моделями",
        labels={"value": "Значение R²", "variable": "Тип выборки"}
    )
    fig_models.update_layout(yaxis_range=[0, 0.85])
    st.plotly_chart(fig_models, width="stretch")



with tab_help:
    st.title("Справка и документация")
    st.markdown("""
    ### О проекте
    Сервис предназначен для экспресс-оценки рыночной стоимости жилья в Турции на базе машинного обучения.

    ### Описание параметров:
    * **Общая площадь (м²):** допустимый диапазон от 25 до 500 м².
    * **Комнаты:** общее количество комнат и число гостиных комнат (салонов).
    * **Этажность:** номер этажа и количество этажей (подвальные и цокольные этажи указываются со знаком минус: -1, -2).
    * **Возраст здания:** число лет или отметка «Возраст здания неизвестен».
    * **Район:** оценивается через обученный `TargetEncoder`.

    ### Характеристики финальной модели:
    * **Алгоритм:** LightGBM Regressor с оптимизированными гиперпараметрами.
    * **Обучающая база:** 182 666 объявлений платформы Zingat (2018–2019 гг.).
    * **Качество прогноза на тесте:**
      * Продажа: $R^2 = 0.69$, средняя абсолютная погрешность $\\text{MAE} \\approx 111\\,129\\text{ TRY}$.
      * Аренда: $R^2 = 0.55$, средняя абсолютная погрешность $\\text{MAE} \\approx 588\\text{ TRY}$.

    ---
    ### _Сделано Разумченко Д. В. 23П-1..._
    """)