Para atingirmos excelência na área de Inteligência Artificial generativa modelos de IA precisam ser desenvolvidos. Esses modelos precisam de sua performance extremamente comprovada em ambiente de produção, neste desafio você vai receber a responsabilidade de avaliar um modelo e afirmar se ele está adequado para uso por usuário reais.

Avaliação de Modelos — Q&A com LLM
Seu objetivo é construir uma pipeline completa para avaliar a capacidade de um modelo em realizar a tarefa de responder perguntas (Q&A) com um LLM usando um conjunto de dados fornecidos.
 Os dados têm as colunas question, content e data_category_QA. Utilize apenas os registros em que data_category_QA ∈ {positivo, negativo}:
positivo: a resposta está contida no content.


negativo: a resposta não está contida no content.

Você pode escolher qualquer modelo (API local ou serviço em nuvem). Caso existam limitações de requisições, selecione um subconjunto representativo dos dados e justifique sua amostragem.
O modelo em questão é de sua escolha, o objetivo é sua solução em descobrir se um modelo tem capacidade de realizar uma tarefa, se o modelo tem desempenho aceitável ou não na tarefa, não é relevante para o teste, por isso utilize o modelo que estiver disponível para você.

Utilize todos meios que achar adequado para chegar a conclusões sobre o modelo escolhido.

Requisitos
Código reprodutível (notebook ou repositório) da pipeline fim a fim.
Relatório conciso com: instruções para executar sua pipeline, procedimento e conclusões sobre as capacidades do modelo na tarefa.
Arquivo CSV com as saídas (por exemplo: id, prediction etc…).
O dataset a ser utilizado é https://huggingface.co/datasets/Weni/WeniEval-Benchmark-2.0.0

Diferencial
Qualquer aspecto adicional que for possível avaliar para enriquecer conclusões em relação ao modelo será um diferencial.
Organização será um diferencial.
Possíveis técnicas de explicabilidade serão um diferencial.

     Recomendação
Documente e justifique impedimentos técnicos que encontrar no caminho.




