
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Dashboard - Projeto Machine Learning PETR4 (2023-2025)")

df = pd.read_csv("PETR4_2023_2025.csv")
df['data'] = pd.to_datetime(df['data'], errors='coerce')
df['ano'] = df['data'].dt.year
df['variacao_pct'] = (df['pre_ult'] - df['pre_abe']) / df['pre_abe']
df['target'] = (df['pre_ult'] > df['pre_abe']).astype(int)

def avaliar_modelo(colunas_features, nome_modelo):
    df_train = df[df['ano'].isin([2023, 2024])]
    df_test = df[df['ano'] == 2025].copy()

    X_train = df_train[colunas_features]
    y_train = df_train['target']
    X_test = df_test[colunas_features]
    y_test = df_test['target']

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    especificidade = tn / (tn + fp)

    df_test['y_pred'] = y_pred
    ganhos = df_test[(df_test['y_pred'] == 1) & (df_test['target'] == 1)]['variacao_pct'].sum()
    perdas = df_test[(df_test['y_pred'] == 1) & (df_test['target'] == 0)]['variacao_pct'].sum()
    saldo = ganhos + perdas

    st.subheader(f"{nome_modelo}")
    st.write(f"**Acurácia:** {acuracia:.2%}")
    st.write(f"**Precisão:** {precisao:.2%}")
    st.write(f"**Recall:** {recall:.2%}")
    st.write(f"**F1-Score:** {f1:.2%}")
    st.write(f"**Especificidade:** {especificidade:.2%}")
    st.write("**💰 Simulação Financeira:**")
    st.write(f"Ganhos: {ganhos:.2%} | Perdas: {perdas:.2%} | Saldo Final: {saldo:.2%}")

st.header("Gráficos do Dataset")
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(df['data'], df['pre_ult'], color='blue')
ax1.set_title("Série Temporal do Preço de Fechamento - PETR4")
ax1.set_xlabel("Data")
ax1.set_ylabel("Preço de Fechamento (R$)")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(5,4))
class_counts = df['target'].value_counts(normalize=True).sort_index() * 100
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', ax=ax2)
ax2.set_title("Distribuição Percentual das Classes")
ax2.set_xlabel("Classe (0=Queda, 1=Alta)")
ax2.set_ylabel("Percentual (%)")
for i, val in enumerate(class_counts.values):
    ax2.text(i, val + 0.5, f"{val:.2f}%", ha='center')
st.pyplot(fig2)

st.header("Resultados dos Modelos")
avaliar_modelo(['pre_abe', 'pre_ult'], "Modelo Bruto (pre_abe, pre_ult)")
avaliar_modelo(['pre_abe', 'variacao_pct'], "Modelo Aprimorado (pre_abe, variacao_pct)")
