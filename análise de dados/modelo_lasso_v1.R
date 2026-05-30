library(dplyr)
library(readr)
library(pROC)
library(ResourceSelection)
library(ggplot2)
library(car)
library(DescTools)

# ==============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==============================================================================
dados <- read_csv("USP/7 semestre/CEA/dados/dados_filtrados.csv")

dados <- dados %>%
  mutate(
    local_aplic_ajustada = ifelse(
      local_aplic_ajustada == "não informado",
      NA_character_,
      local_aplic_ajustada
    )
  )
dados <- dados %>%
  mutate(
    gestante = relevel(as.factor(gestante), ref = "Nao Informado"))
levels(dados$gestante)

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
    pais_nasc = relevel(as.factor(pais_nasc), ref = "nao informado"))
levels(dados$pais_nasc)

dados <- dados %>%
  mutate(
    idade_anos = as.factor(idade_anos))

variavel_resposta    <- "cls_fin_reg_logistica"
variavel_explicativa <- c(
  "dt_not_ano", "dose_ajustada", "regiao", "sexo",
  "idade_anos",  "local_aplic_ajustada"
)

dados[[variavel_resposta]] <- as.factor(dados[[variavel_resposta]])

dados_modelo <- dados %>%
  select(all_of(c(variavel_resposta, variavel_explicativa))) %>%
  na.omit()

cat("N após na.omit:", nrow(dados_modelo), "\n")
cat("Distribuição da resposta:\n")
print(prop.table(table(dados_modelo[[variavel_resposta]])))

# ==============================================================================
# 2. MODELO LOGÍSTICO — TODOS OS DADOS
# Objetivo: inferência, não predição
# Não há divisão treino/teste
# ==============================================================================
formula_modelo <- as.formula(paste(variavel_resposta, "~ ."))

modelo <- glm(
  formula_modelo,
  data   = dados_modelo,
  family = binomial
)

cat("\n--- SUMÁRIO DO MODELO ---\n")
print(summary(modelo))

# ==============================================================================
# 3. ODDS RATIOS COM IC 95% — resultado principal
# ==============================================================================
cat("\n--- ODDS RATIOS (IC 95%) ---\n")
or_tabela <- exp(cbind(
  OR     = coef(modelo),
  confint(modelo)
))
print(round(or_tabela, 4))

# ==============================================================================
# 4. TABELA ORGANIZADA DE OR (para relatório)
# ==============================================================================
or_df <- as.data.frame(round(or_tabela, 4))
colnames(or_df) <- c("OR", "IC_2.5%", "IC_97.5%")
or_df$p_valor <- round(summary(modelo)$coefficients[, 4], 4)
or_df$significativo <- ifelse(or_df$p_valor < 0.05, "Sim", "Não")
or_df <- or_df %>% filter(rownames(or_df) != "(Intercept)")

cat("\n--- TABELA FINAL DE OR ---\n")
print(or_df)

# ==============================================================================
# 5. DIAGNÓSTICOS DO MODELO
# ==============================================================================

# VIF — multicolinearidade
cat("\n--- VIF ---\n")
print(round(vif(modelo), 4))

# Pseudo R² — múltiplos
cat("\n--- PSEUDO R² ---\n")
todos_r2 <- PseudoR2(modelo,
                     which = c("McFadden", "McFaddenAdj", "CoxSnell", "Nagelkerke"))
print(round(todos_r2, 4))

# Tjur R²
prob_todos  <- predict(modelo, type = "response")
y_num_todos <- as.numeric(dados_modelo[[variavel_resposta]]) - 1
tjur_r2 <- mean(prob_todos[y_num_todos == 1]) -
  mean(prob_todos[y_num_todos == 0])
cat("Tjur R²:", round(tjur_r2, 4), "\n")

# AIC e BIC
cat("\nAIC:", round(AIC(modelo), 2))
cat("\nBIC:", round(BIC(modelo), 2), "\n")

# Hosmer-Lemeshow — calibração
cat("\n--- HOSMER-LEMESHOW ---\n")
print(hoslem.test(y_num_todos, prob_todos, g = 10))

# Brier Score
cat("\n--- BRIER SCORE ---\n")
cat("Valor:", round(mean((prob_todos - y_num_todos)^2), 4), "\n")

# ==============================================================================
# 6. CURVA ROC E AUC
# ==============================================================================
curva_roc <- roc(response  = as.numeric(dados_modelo[[variavel_resposta]]),
                 predictor = prob_todos,
                 quiet     = TRUE)
auc_valor <- auc(curva_roc)
cat("\nAUC:", round(auc_valor, 4), "\n")

print(
  ggroc(curva_roc, legacy.axes = TRUE, color = "#008080", size = 1.2) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "grey60")  +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title    = "Curva ROC — Regressão Logística Inferencial",
      subtitle = paste0("AUC: ", round(auc_valor * 100, 1), "%"),
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