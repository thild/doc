
# Knn com doc2vec

Trabalho da disciplina de Aprendizagem de Máquina

# Objetivo

Implementar o classificador KNN para a análise de sentimento da base IMDB

# Metodologia

Primeiramento foi realizado o pré-processamento do conjunto de dados.
 
Foi escolhido o dataset de http://ai.stanford.edu/%7Eamaas/data/sentiment/ (http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz) que estava com um formato de arquivo mais fácil para pré-processar.

Foram extraídos todos os caracteres não alfabéticos, as palavras com menos de três caracteres, stopwords e substantivos.

As características foram extraídas usando o algoritmos doc2vec da biblioteca gensim com os seguintes parâmetros:

```  Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=16) ```

O modelo foi treinado durante 20 épocas.

O KNN foi testado com diversos valores de k e de tamanho de dataset, porém não foram realizados testes com a base inteira devido ao tempo de processamento elevado.

Foi testada apenas a distância euclidiana.

# Resultados

A implementação do knn é naive, sem otimizações, escrita em python. Optou-se por python para aprendizado da linguagem.

Os resultados obtidos foram comparados com a implementação do KNN da biblioteca scikit-learn. O desempenho do algoritmo implementado é bastante inferior que a implementação do scikit, o que era esperado, pois não foram realizadas otimziações. Os resultados de score foram idênticos aos do scikit.

O melhor K encontrado nos testes empíricos foi 9.

O melhor score obtido foi 76% com 5000 instâncias de treinamento e 5000 instâncias de teste.

Ainda não foram realizados testes com os dados não-rotulados.

# Considerações finais

O desempenho do algoritmo KNN com implementação sem otimizações é ineficiente. Como melhoria propõe-se a implementação com partição binária que reduz a complexidade para log n.

Também seria interessante testar mais a extração de características, alterando os parâmetros do doc2vec. Aparentemente a remoção dos substantivos não afetou consideravelmente a taxa de acertos, porém são necessários testes mais aprofundados.


Copyright (c) 2015 [Tony Alexander Hild](https://github.com/thild)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
