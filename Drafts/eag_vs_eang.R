library(dplyr)
library(readr)
library(pROC)
library(caret)
library(ggplot2)
library(car)
library(ResourceSelection)
library(DescTools)

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS (100% DA AMOSTRA)
# ==============================================================================
dados <- read_csv("USP/7 semestre/CEA/dados/dados_eang_vs_eag.csv")

# Criando a variável combinada de manifestações e ajustando fatores
dados <- dados %>%
  mutate(
    manifestacoes = if_else(
      manifestacoes_locais == 1 | manifestacoes_sistemicas == 1, 1, 0
    ),
    idade_anos = as.factor(idade_anos),
    dt_apl_ano = as.factor(dt_apl_ano)
  )

dados <- dados %>%
  mutate(
    perfil_biologico = case_when(
      sexo == "Masculino" ~ "Homem",
      sexo == "Feminino" & gestante == "Sim" ~ "Mulher Gestante",
      sexo == "Feminino" & gestante == "Não" ~ "Mulher Não Gestante",
      TRUE ~ "Não Informado"
    ),
    perfil_biologico = as.factor(perfil_biologico)
  )

dados <- dados %>%
  mutate(
    via_adm_ajustada = relevel(as.factor(via_adm_ajustada), ref = "vazio"))
levels(dados$via_adm_ajustada)
variavel_resposta    <- "cls_fin_reg_logistica_eag_eang"
variavel_explicativa <- c(
  "regiao", "dose_ajustada", "dt_not_ano",
  "via_adm_ajustada", "idade_anos",
  "manifestacoes_locais", "manifestacoes_sistemicas"
)

# Ajustando a classe de referência (FALSE como base)
dados[[variavel_resposta]] <- relevel(
  as.factor(dados[[variavel_resposta]]), ref = "FALSE"
)

# Criando o dataset final (sem NA's)
dados_modelo <- dados %>%
  select(all_of(c(variavel_resposta, variavel_explicativa))) %>%
  na.omit()

cat("Dimensão total para Inferência:", nrow(dados_modelo), "linhas\n")
cat("Distribuição da resposta (Realidade Clínica):\n")
print(prop.table(table(dados_modelo[[variavel_resposta]])))

# ==============================================================================
# 2. MODELO LOGÍSTICO INFERENCIAL (N = 100%)
# ==============================================================================
formula_inferencia <- as.formula(paste(variavel_resposta, "~ ."))

modelo_logistico <- glm(
  formula_inferencia,
  data   = dados_modelo,
  family = binomial
)

cat("\n--- SUMÁRIO DO MODELO (COEFICIENTES E P-VALORES REAIS) ---\n")
print(summary(modelo_logistico))

# ==============================================================================
# 3. RAZÃO DE CHANCES (ODDS RATIO) COM IC 95%
# ==============================================================================
cat("\n--- ODDS RATIOS (IC 95%) ---\n")
# Usamos suppressMessages pois o confint pode avisar sobre perfis de verossimilhança
suppressMessages({
  or_tabela <- exp(cbind(OR = coef(modelo_logistico), confint(modelo_logistico)))
})
print(round(or_tabela, 4))

# ==============================================================================
# 5. CURVA ROC E AUC APARENTE (ESTATÍSTICA C)
# ==============================================================================
probabilidades <- predict(modelo_logistico, type = "response")
y_real <- dados_modelo[[variavel_resposta]]
curva_roc <- roc(response  = y_real,
                 predictor = probabilidades,
                 levels    = c("FALSE", "TRUE"),
                 direction = "<",
                 quiet     = TRUE)

auc_valor <- auc(curva_roc)
cat("\nAUC Aparente (Estatística C):", round(auc_valor, 4), "\n")

# Gráfico ROC
print(
  ggroc(curva_roc, legacy.axes = TRUE, color = "#008080", size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title    = "Curva ROC — Qualidade de Ajuste do Modelo Explicativo",
      subtitle = paste0("Estatística C (AUC): ", round(auc_valor * 100, 1), "%"),
      x        = "Taxa de Falso Positivo (1 - Especificidade)",
      y        = "Sensibilidade"
    ) +
    theme_minimal() +
    theme(
      plot.title    = element_text(face = "bold", size = 14, color = "#003366"),
      plot.subtitle = element_text(size = 12, color = "grey40")
    )
)

# ==============================================================================
# 6. CALIBRAÇÃO (HOSMER-LEMESHOW)
# ==============================================================================
y_num_real <- as.numeric(y_real == "TRUE")

cat("\n--- HOSMER-LEMESHOW TEST (CALIBRAÇÃO) ---\n")
print(hoslem.test(y_num_real, probabilidades, g = 10))

brier_score <- mean((probabilidades - y_num_real)^2)
cat("\n--- BRIER SCORE ---\n")
cat("Valor:", round(brier_score, 4), "\n")

# ==============================================================================
# 7. PSEUDO R² E CRITÉRIOS DE INFORMAÇÃO
# ==============================================================================
todos_r2 <- PseudoR2(modelo_logistico, 
                     which = c("McFadden", "McFaddenAdj", "CoxSnell", "Nagelkerke"))

tjur_r2 <- mean(probabilidades[y_num_real == 1]) - mean(probabilidades[y_num_real == 0])

cat("\n--- PSEUDO R² ---\n")
cat("McFadden:          ", round(todos_r2["McFadden"]    * 100, 2), "%\n")
cat("McFadden Ajustado: ", round(todos_r2["McFaddenAdj"] * 100, 2), "%\n")
cat("Cox-Snell:         ", round(todos_r2["CoxSnell"]    * 100, 2), "%\n")
cat("Nagelkerke:        ", round(todos_r2["Nagelkerke"]  * 100, 2), "%\n")
cat("Tjur R²:           ", round(tjur_r2                 * 100, 2), "%\n")

cat("\nAIC:", round(AIC(modelo_logistico), 2), "\n")
cat("BIC:", round(BIC(modelo_logistico), 2), "\n")

# ==============================================================================
# 8. MULTICOLINEARIDADE (VIF)
# ==============================================================================
cat("\n--- VIF (DIAGNÓSTICO DE MULTICOLINEARIDADE) ---\n")
print(round(vif(modelo_logistico), 4))


# ==============================================================================
# REGRESSÃO LOGÍSTICA SIMPLES (UNIVARIADA) PARA CADA VARIÁVEL
# ==============================================================================

# Criando uma lista vazia para armazenar os resultados
resultados_univariados <- list()

cat("Iniciando regressões univariadas...\n")

for (var in variavel_explicativa) {
  
  # 1. Cria a fórmula dinâmica: ex: cls_fin_reg_logistica ~ sexo
  formula_simples <- as.formula(paste(variavel_resposta, "~", var))
  
  # 2. Roda o modelo simples
  modelo_simples <- glm(formula_simples, data = dados_modelo, family = binomial)
  
  # 3. Extrai os coeficientes e p-valores
  resumo <- summary(modelo_simples)$coefficients
  
  # 4. Calcula OR e IC 95% (suprimindo os avisos normais de verossimilhança)
  suppressMessages({
    ic <- exp(confint(modelo_simples))
  })
  
  # 5. Monta uma tabela temporária apenas para as categorias da variável atual
  # Ignoramos a linha 1 que é sempre o "(Intercept)"
  tabela_temp <- data.frame(
    Variavel_Categoria = rownames(resumo)[-1],
    OR_Bruto         = exp(resumo[-1, "Estimate"]),
    IC_2.5           = ic[-1, 1],
    IC_97.5          = ic[-1, 2],
    p_valor          = resumo[-1, "Pr(>|z|)"]
  )
  
  # 6. Adiciona a indicação de significância
  tabela_temp$Significativo <- ifelse(tabela_temp$p_valor < 0.05, "Sim", "Não")
  
  # 7. Guarda na nossa lista
  resultados_univariados[[var]] <- tabela_temp
}

# Junta todos os pedaços da lista em um único Data Frame final
tabela_final_univariada <- do.call(rbind, resultados_univariados)
rownames(tabela_final_univariada) <- NULL # Limpa o índice visual

# Arredonda os números para ficar igual ao seu padrão de visualização
tabela_final_univariada <- tabela_final_univariada %>%
  mutate(
    OR_Bruto = round(OR_Bruto, 4),
    IC_2.5   = round(IC_2.5, 4),
    IC_97.5  = round(IC_97.5, 4),
    p_valor  = round(p_valor, 4)
  )

cat("\n--- RESULTADOS: REGRESSÃO LOGÍSTICA SIMPLES (ODDS RATIO BRUTO) ---\n")
print(tabela_final_univariada)