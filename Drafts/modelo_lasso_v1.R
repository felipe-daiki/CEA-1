# Carregando os pacotes necessários
library(dplyr)
library(readr)
library(glmnet)
library(pROC)
library(caret)
library(ResourceSelection)

# Carrega os dados
dados <- read_csv("USP/7 semestre/CEA/dados/dados_filtrados.csv")

variavel_resposta <- "cls_fin_reg_logistica"

variavel_explicativa <- c(
  "dt_not_ano", "dose_ajustada",
  'sexo', 'regiao',
  "via_adm_ajustada",
  "tp_atd_ajustada", "idade_anos"
)

#variavel_explicativa <- c(
#  "dt_not_ano", "regiao", "pais_nasc", "sexo", "cor", 
#  "gestante", "cod_prod_ajustada", "dt_apl_ano", "dose_ajustada", 
#  "via_adm_ajustada", "local_aplic_ajustada", "tp_med", 
#  "tp_atd_ajustada", "idade_anos", "cls_compl_ajustada", 
#  "dor_abdominal", "dor_no_corpo", "artralgia", "cefaleia", "dor", 
#  "exantema_local", "exantema", "edema", "eritema", "calor", 
#  "enduracao", "abscesso_quente", "lesao", "linfonodomegalia", 
#  "prurido", "febre", "nausea", "emese", "diarreia", "tontura", 
#  "sincope", "parestesia", "convulsao", "confusao_mental", 
#  "fraqueza", "hipotensao", "taquicardia", "bradicardia", 
#  "extremidades_frias", "palidez", "sudorese", "urticaria", 
#  "broncoespasmo", "dispneia", "angioedema", "tremor", 
#  "fotofobia", "visao_turva", "guillain_barre", "encefalite", 
#  "epilepsia", "paralisia", "purpura_trombocitopenica", 
#  "manifestacoes_locais", "manifestacoes_sistemicas"
#)


# Transforma a resposta em Fator
dados[[variavel_resposta]] <- as.factor(dados[[variavel_resposta]])

# FILTRO: Mantém apenas as variáveis de interesse
colunas_para_modelo <- c(variavel_resposta, variavel_explicativa)
dados_lasso <- dados %>% select(all_of(colunas_para_modelo))

# Remove linhas com NAs (dados faltantes) apenas nessas colunas, pois o glmnet não aceita NA
dados_lasso <- na.omit(dados_lasso)

# Matriz X e Vetor Y
# ANTES (Sem interação):
# x <- model.matrix(as.formula(paste(variavel_resposta, "~ .")), data = dados_lasso)[ , -1]

# DEPOIS (Com todas as interações duplas):
x <- model.matrix(as.formula(paste(variavel_resposta, "~ .^2")), data = dados_lasso)[ , -1]
y <- dados_lasso[[variavel_resposta]]

# ==============================================================================
# 2. TREINAMENTO COM VALIDAÇÃO CRUZADA (CROSS-VALIDATION)
# ==============================================================================
set.seed(42) 
modelo_lasso_cv <- cv.glmnet(x, y, family = "binomial", alpha = 1)

melhor_lambda <- modelo_lasso_cv$lambda.min
cat("O melhor valor de penalidade (Lambda) encontrado foi:", round(melhor_lambda, 4), "\n\n")

# ==============================================================================
# 3. EXTRAINDO O RESULTADO FINAL E AS VARIÁVEIS SOBREVIVENTES
# ==============================================================================
coeficientes_finais <- coef(modelo_lasso_cv, s = "lambda.min")

tabela_coeficientes <- as.matrix(coeficientes_finais)
tabela_coeficientes <- data.frame(
  Variavel = rownames(tabela_coeficientes),
  Coeficiente = tabela_coeficientes[, 1]
)

variaveis_sobreviventes <- tabela_coeficientes %>%
  filter(Coeficiente != 0) %>%
  arrange(desc(abs(Coeficiente))) 

rownames(variaveis_sobreviventes) <- NULL

print("--- VARIÁVEIS SELECIONADAS PELO LASSO ---")
print(variaveis_sobreviventes)

# ==============================================================================
# OPCIONAL: GRÁFICO DA PENALIZAÇÃO
# ==============================================================================
plot(modelo_lasso_cv)

# ==============================================================================
# 1. GERANDO AS PREVISÕES DO MODELO
# ==============================================================================
# O modelo vai olhar para os pacientes (matriz x) e tentar adivinhar o desfecho.

# Extraindo a probabilidade (de 0% a 100%)
probabilidades <- predict(modelo_lasso_cv, newx = x, s = "lambda.min", type = "response")
probabilidades <- as.numeric(probabilidades) # Converte para vetor numérico simples

# Extraindo a classificação final (qual categoria o modelo escolheu)
classes_previstas <- predict(modelo_lasso_cv, newx = x, s = "lambda.min", type = "class")
classes_previstas <- as.factor(as.vector(classes_previstas))

# ==============================================================================
# 2. MATRIZ DE CONFUSÃO E MÉTRICAS CLÁSSICAS
# ==============================================================================
# Garante que os níveis dos fatores (levels) sejam os mesmos para não dar erro
classes_previstas <- factor(classes_previstas, levels = levels(y))

matriz_confusao <- confusionMatrix(data = classes_previstas, reference = y)

print("--- 1. MATRIZ DE CONFUSÃO E MÉTRICAS ---")
print(matriz_confusao)

# ==============================================================================
# 3. CURVA ROC E AUC (DISCRIMINAÇÃO)
# ==============================================================================
# Mede a capacidade do modelo de separar quem tem o evento grave de quem não tem.
curva_roc <- roc(response = as.numeric(y), predictor = probabilidades, quiet = TRUE)
auc_valor <- auc(curva_roc)

cat("\n--- 2. AUC (Area Under the Curve) ---\n")
cat("Valor da AUC:", round(auc_valor, 4), "\n")

# Instale o pacote se não tiver:
# install.packages("ggplot2")
library(ggplot2)
library(pROC) # Garante que pROC está carregado para usar ggroc

# ==============================================================================
# PROPOSTA: Gráfico da Curva ROC (ggplot2)
# ==============================================================================

# 1. Preparação: Converter o objeto pROC em dados compatíveis com ggplot2
# Usamos legacy.axes = TRUE para plotar Sensibilidade vs (1 - Especificidade)
df_roc <- ggroc(curva_roc, legacy.axes = TRUE)

# Obter as coordenadas para personalizar o gráfico manualmente (opcional, mas bom)
coords_roc <- coords(curva_roc, "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
coords_roc$`1-Specificity` <- 1 - coords_roc$specificity

# 2. Criando o Plot com ggplot2
plot_bonito <- ggplot(coords_roc, aes(x = `1-Specificity`, y = sensitivity)) +
  # Adiciona a linha da Curva ROC (usando sua cor teal #008080)
  geom_line(color = "#008080", size = 1.5) +
  
  # Adiciona a linha de referência diagonal (tracejada, indicando o acaso)
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60", size = 0.8) +
  
  # Adiciona uma sombra suave sob a curva (estética opcional, mas fica bonito)
  geom_area(fill = "#008080", alpha = 0.1) +
  
  # Configuração dos Eixos (forçando ir de 0 a 1 e formatando como porcentagem)
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1), labels = scales::percent) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1), labels = scales::percent) +
  
  # Rótulos e Títulos
  labs(
    title = "Gráfico da Curva ROC",
    subtitle = paste0("Modelo LASSO - Área Sob a Curva (AUC): ", round(auc_valor * 100, 2), "%"),
    x = "Taxa de Falso Positivo (1 - Especificidade)",
    y = "Taxa de Verdadeiro Positivo (Sensibilidade)"
  ) +
  
  # Aplicação de um Tema Limpo e Profissional
  theme_minimal() +
  theme(
    # Centraliza e formata títulos
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, color = "#003366"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey30", margin = margin(b = 15)),
    
    # Formata rótulos dos eixos
    axis.title = element_text(face = "bold", size = 11),
    axis.text = element_text(size = 10),
    
    # Adiciona borda leve e remove grades excessivas
    panel.border = element_rect(color = "grey80", fill = NA, size = 1),
    panel.grid.minor = element_blank(),
    
    # Ajusta margens
    plot.margin = margin(20, 20, 20, 20)
  )

# 3. Visualizar o gráfico na tela
print(plot_bonito)

# ==============================================================================
# 4. TESTE DE HOSMER-LEMESHOW (CALIBRAÇÃO / GOODNESS-OF-FIT)
# ==============================================================================
# O pacote ResourceSelection exige que o 'y' real seja 0 ou 1 numérico
y_numerico <- as.numeric(y) - 1 

# g = 6 foi escolhido devido ao seu baixo número de casos raros
hl_teste <- hoslem.test(y_numerico, probabilidades, g = 5)

print("\n--- 3. TESTE DE HOSMER-LEMESHOW ---")
print(hl_teste)

# ==============================================================================
# 5. BRIER SCORE (Acurácia das Probabilidades)
# ==============================================================================
# O y_numerico já foi criado no seu código (0 ou 1)
brier_score <- mean((probabilidades - y_numerico)^2)

cat("\n--- 4. BRIER SCORE ---\n")
cat("Valor do Brier Score:", round(brier_score, 4), "\n")
cat("(Valores mais próximos de 0 indicam previsões probabilísticas melhores)\n")

# ==============================================================================
# 6. PSEUDO R² (Deviance Ratio)
# ==============================================================================
# No glmnet, podemos extrair a proporção do 'Deviance' explicada pelo modelo
# (semelhante ao R² da regressão linear)
indice_lambda_min <- which(modelo_lasso_cv$lambda == modelo_lasso_cv$lambda.min)
pseudo_r2 <- modelo_lasso_cv$glmnet.fit$dev.ratio[indice_lambda_min]

cat("\n--- 5. PSEUDO R² (DEVIANCE RATIO) ---\n")
cat("Proporção da variância explicada (Pseudo R²):", round(pseudo_r2 * 100, 2), "%\n")

# ==============================================================================
# 7. GRÁFICO DE CALIBRAÇÃO (Visualizando o ajuste)
# ==============================================================================
library(dplyr)
library(ggplot2)

# Criando decis de probabilidade para o gráfico
df_calibracao <- data.frame(
  Observado = y_numerico,
  Previsto = probabilidades
) %>%
  mutate(Decil = ntile(Previsto, 10)) %>% # Divide em 10 grupos de risco
  group_by(Decil) %>%
  summarise(
    Media_Prevista = mean(Previsto),
    Taxa_Observada = mean(Observado),
    N = n()
  )

plot_calibracao <- ggplot(df_calibracao, aes(x = Media_Prevista, y = Taxa_Observada)) +
  geom_point(color = "#003366", size = 3) +
  geom_line(color = "#008080", size = 1) +
  # Linha de calibração perfeita (diagonal)
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title = "Gráfico de Calibração do Modelo LASSO",
    subtitle = "Comparação entre Probabilidade Prevista e Taxa Real Observada",
    x = "Probabilidade Prevista Média (por Decil)",
    y = "Taxa Real de Eventos Observada"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, color = "#003366"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "grey30", margin = margin(b = 15)),
    panel.border = element_rect(color = "grey80", fill = NA, size = 1)
  )

print(plot_calibracao)

# ==============================================================================
# TESTE DE MULTICOLINEARIDADE (INDEPENDÊNCIA DAS VARIÁVEIS)
# ==============================================================================
# Instale o pacote 'car' se ainda não o tiver:
# install.packages("car")
library(car)

cat("\n========================================\n")
cat("TESTE DE MULTICOLINEARIDADE (VIF)\n")
cat("========================================\n")

# Para testar a multicolinearidade, rodamos um modelo logístico clássico
# (sem penalização LASSO e sem as interações ^2, apenas os efeitos principais)
formula_vif <- as.formula(paste(variavel_resposta, "~ ."))
modelo_classico <- glm(formula_vif, data = dados_lasso, family = binomial)

# Calculando o VIF
# Nota: Se tiver variáveis categóricas com mais de 2 níveis, o 'car' calcula o GVIF (Generalized VIF)
valores_vif <- vif(modelo_classico)

print(valores_vif)

