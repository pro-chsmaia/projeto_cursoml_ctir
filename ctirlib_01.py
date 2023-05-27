# CTir - Classificação Textual com incidência de ruído
## Funções auxiliares

def dividirTreinoTeste(df, dfQtLinhas):
    from sklearn.model_selection import train_test_split
    
    # Seleção de linhas do DataFrame
    dfSelecao = df.sample(n=dfQtLinhas, random_state=42)
    
    # Variável dependente e independente (alvo)
    X = dfSelecao.texto
    y = dfSelecao.classe
    
    # dividir dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
       
    return X_treino, X_teste, y_treino, y_teste

def gerarMatrizTfidf(X_treino, X_teste, minimoDf=1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # matriz tfidf
    vetorizador_tfidf = TfidfVectorizer(min_df=minimoDf)
    X_treino_tfidf    = vetorizador_tfidf.fit_transform(X_treino)
    X_teste_tfidf     = vetorizador_tfidf.transform(X_teste)

    return vetorizador_tfidf, X_treino_tfidf, X_teste_tfidf

def treinarModeloSGD(X_treino_tfidf, y_treino, X_teste_tfidf, y_teste, sgd_loss, retornaClassificador=False):
    from sklearn.linear_model import SGDClassifier

    # Stochastic Gradient Descent
    sgdModelo = SGDClassifier(loss=sgd_loss, random_state=42)
    sgdModelo.fit(X_treino_tfidf, y_treino)
    
    # Acurácia %
    acuraciaSGD = round(sgdModelo.score(X_teste_tfidf, y_teste) * 100, 2)
    retorno = acuraciaSGD
    if retornaClassificador:
        retorno = [acuraciaSGD, sgdModelo]

    return retorno

def gerarDfMedidasAcuracia(listaAcuraciasDfClasseTexto, listaAcuraciasDfClasseTextoRuido):
    import pandas as pd
    
    listaMedidasAcuracia = []
    
    for i in range(len(listaAcuraciasDfClasseTexto)):
        # SEM ruído
        listaMedidasAcuracia.append([listaAcuraciasDfClasseTexto[i][0], \
                                     listaAcuraciasDfClasseTexto[i][1], \
                                     listaAcuraciasDfClasseTexto[i][2], \
                                     False,listaAcuraciasDfClasseTexto[i][3], \
                                     listaAcuraciasDfClasseTexto[i][4]])
        
    for i in range(len(listaAcuraciasDfClasseTextoRuido)):
        # COM ruído
        listaMedidasAcuracia.append([listaAcuraciasDfClasseTextoRuido[i][0], \
                                     listaAcuraciasDfClasseTextoRuido[i][1], \
                                     listaAcuraciasDfClasseTextoRuido[i][2], \
                                     True,listaAcuraciasDfClasseTextoRuido[i][3], \
                                     listaAcuraciasDfClasseTextoRuido[i][4]])

    dfMedidasAcuracia = pd.DataFrame(listaMedidasAcuracia, columns=['qtLinhas', 'acuracia', \
                                                                    'sgd_loss', 'ruido', 'mindf', 'tfidf_limitado'])
    
    return dfMedidasAcuracia

def graficoEvolucaoAcuracia(dfMedidasAcuracia, titulo, \
                            separarTiposClassificadores, \
                            separarMinDf, \
                            separarTfidfLimitado, \
                            exibirSemRuido, exibirComRuido, eixoXmaximo = 10000):    
    """
        Gráfico de evolução da acurácia
        ---
        ### Parâmetros
        :dfMedidasAcuracia: DataFrame com os campos
                            qtLinhas, acuracia, sgd_loss, ruido
        :titulo           : Título do gráfico.
        :separarTiposClassificadores: agrupar por classificador (sgd_loss)
        :separarMinDf     : agrupar por mínimo df (min_df)
        :separarTfidfLimitado: agrupar Tfidf limitado
        :exibirSemRuido   : exibir dados sem ruído
        :exibirComRuido   : exibir dados com ruído
        ### Retorno
        :return: figura do gráfico
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    import pandas  as pd
    
    ## Estilo
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    
    sns.set_theme(style="ticks", palette="deep", rc=custom_params)
    fig = plt.figure(figsize=(7, 4))
    
    plt.suptitle('Classificação Textual (incidência de ruído)')
    plt.title(titulo)
    plt.grid()
    
    ## Eixos - escala e label
    plt.xlabel('Quantidade de linhas do DataFrame')
    plt.ylabel('Acurácia (%)')
    
    intervaloEixoX = int(eixoXmaximo/20)
    plt.xticks(range(0, eixoXmaximo+1, intervaloEixoX), fontsize=6)
    plt.yticks(range(0, 101, 5), fontsize=6)

    plt.xlim([0, eixoXmaximo])
    plt.ylim([50, 100])
    
    ### Gráficos de linha 
    if separarTiposClassificadores:
        lpAcuraciaTiposClassif = sns.lineplot(x='qtLinhas', y='acuracia',\
                                              data=dfMedidasAcuracia, linewidth=2,\
                                              markers=True, hue='sgd_loss', style='ruido')
    elif separarMinDf:
        lpAcuraciaMinDf        = sns.lineplot(x='qtLinhas', y='acuracia',\
                                              data=dfMedidasAcuracia, linewidth=1, markers=True, hue='mindf')
    elif separarTfidfLimitado:
        lpAcuraciaMinDf        = sns.lineplot(x='qtLinhas', y='acuracia',\
                                              data=dfMedidasAcuracia, linewidth=2, markers=True, hue='tfidf_limitado',\
                                              style='tfidf_limitado')
    else:        
        if (exibirSemRuido):
            dfMedidasAcuraciaSemRuido = dfMedidasAcuracia[dfMedidasAcuracia.ruido == False]
            lpAcuracia = sns.lineplot(x='qtLinhas', y='acuracia',\
                                      data=dfMedidasAcuraciaSemRuido, markers=True, label='dados SEM ruído')

        if (exibirComRuido):
            dfMedidasAcuraciaComRuido = dfMedidasAcuracia[dfMedidasAcuracia.ruido]
            lpAcuraciaRuido  = sns.lineplot(x='qtLinhas', y='acuracia',\
                                            data=dfMedidasAcuraciaComRuido, linestyle="dashed", linewidth=2, \
                                            markers=True, label='dados COM ruído')

    return fig

def graficoProporcaoTreinoTeste(y_treino, y_teste, titulo):
    """
        Gráfico de proporção dos dados de
        treino e teste.
        ---
        ### Parâmetros
        :y_treino: variável dependente dos
                   dados de treino.
        :y_teste : variável dependente dos
                   dados de teste.
        :titulo  : título do gráfico.
        ### Retorno
        :return: figura o gráfico
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas  as pd
    
    proporcaoDadosTreino = y_treino.value_counts(normalize=True)
    proporcaoDadosTeste  = y_teste.value_counts(normalize=True)
    dfTreino = pd.DataFrame({'proporcao': proporcaoDadosTreino, 'tipo':'treino'})
    dfTeste  = pd.DataFrame({'proporcao': proporcaoDadosTeste, 'tipo':'teste'})

    dfTreinoTeste = pd.concat([dfTreino, dfTeste])
    dfTreinoTeste.reset_index(inplace=True)
    dfTreinoTeste.rename(columns={'index':'classe'}, inplace=True)    
    
    fig = plt.figure(figsize=(6, 5), facecolor='white')
    fig.suptitle('Proporção dos dados de treino e de teste')
    plt.title(titulo)
    plt.grid()
    
    
    # Gráfico de barras
    bpProporcaoTreinoTeste  = sns.barplot(x='proporcao', y='tipo', data=dfTreinoTeste, hue='classe', palette="pastel")
    bpProporcaoTreinoTeste.legend(ncol=1, loc='lower right', frameon=True, fontsize=9);  

    bpProporcaoTreinoTeste.set_xlabel('Proporção')
    bpProporcaoTreinoTeste.set_ylabel('Classe')

    return fig
