import streamlit as st
import pandas as pd
import time
from transformers import pipeline

# Tytuł aplikacji
st.title('Eksplorator NLP: Analiza Wydźwięku i Tłumaczenie')

# Nagłówki
st.header('Wprowadzenie do zajęć')
st.subheader('O Streamlit')
st.text('To przykładowa aplikacja z wykorzystaniem Streamlit')

st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')

# Ładowanie pliku CSV
df = pd.read_csv("DSP_4.csv", sep=';')
st.dataframe(df)

st.header('Przetwarzanie języka naturalnego')

# Wybór opcji
option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu (eng -> ger)"
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner("Analizuję wydźwięk emocjonalny..."):
            try:
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                st.success("Analiza zakończona pomyślnie!")
                st.write(answer)
            except Exception as e:
                st.error(f'Błąd podczas analizy')
elif option == "Tłumaczenie tekstu (eng -> ger)":
    text = st.text_area(label="Wpisz tekst po angielsku")
    if text:
        with st.spinner("Tłumaczę..."):
            try:
                translator = pipeline("translation_en_to_de")
                translation = translator(text)
                st.success("Tłumaczenie zakończone pomyślnie!")
                st.write(translation[0]['translation_text'])
            except Exception as e:
                st.error(f'Błąd podczas tłumaczenia')

st.subheader('Instrukcja użytkowania')
st.write("""
Aplikacja służy do analizy wydźwięku emocjonalnego tekstu oraz tłumaczenia tekstu z języka angielskiego na niemiecki.
- Wybierz odpowiednią opcję z listy.
- Wpisz tekst do analizy lub tłumaczenia.
- Poczekaj na wynik działania modelu.
""")

st.subheader('Informacje dodatkowe')
st.write('Numer indeksu: 22176')
st.write('[Repozytorium](https://github.com/d3ska/SUML)')
