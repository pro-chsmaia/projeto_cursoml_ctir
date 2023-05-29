## CTir - Classificação Textual em documentos com relevante incidência de ruído
![Em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=blue)
![Projeto_didatico](http://img.shields.io/static/v1?label=FINALIDADE&message=DIDÁTICA&color=green)

### Projeto didático apresentado no Curso:
***Inteligência Artificial na prática: Machine Learning*** / ESMPU - *Maio/2023*

---

Equipe:
> Christiano Maia
>
> [Denard Soares](https://www.linkedin.com/in/denard)

### Desafio
- Recuperação de acurácia em classificação de textos com alta incidência de ruído.
  - Erros no reconhecimento óptico de caracteres (OCR);
  - Erros de digitação (como substituição ou exclusão aleatória de caracteres em palavras).

### Requisitos da base de dados
- Mínimo de 10K linhas rotuladas.

- Campos:
> 'classe': categoria do texto
> 
> 'texto' : texto sem ruído
> 
> 'textoComRuido': texto após aumento de dados (ruído)


### Experimentos implementados

#### SGD Classifier loss='log_loss' sobre texto SEM e COM ruído
> Queda sensível para dados COM ruído;
> 
> Recuperação progressiva da acurácia em função do aumento da quantidade de dados.

#### SGD Classifier loss='log_loss' com mind_df = 4
> Cut-off: mínimo DF (document frequency);
> 
> Melhoria ínfima e apenas com aumento da quantidade de dados.

#### SGD Classifier loss='log_loss' com Tf-idf com limitadores
> Cut-off para reduzir ou eliminar tokens de baixa relevância (ruído);
> 
> Potencializar tokens de alta relevância.

#### SGD Classifier loss='log_loss' e loss='hinge' sobre texto COM ruído
> Princípio da dobradiça (hinge) de margem máxima do SVM
> 
> Melhora progressiva da acurácia e aparente estabilização.

#### Resumo: SGD Classifier loss='hinge', min_df=4 e Tf-idf limitado
