# data analysis rust
# load data
library(readxl)
source("https://raw.githubusercontent.com/Cassava2050/PPD/main/utilities_tidy.R")
library(asreml)
coffee <- read_excel("source_data/IMLVTRustGlobalPerYear_V01102024.xlsx")

plant_date <- read_excel("source_data/planting_dates.xlsx") %>% select(-num)

colnames(coffee) <- c("country", "Site", "site", "variety", "block", "RustScore", "year")

coffee <- coffee %>% select(-c(country, Site)) %>%
  group_by(site, variety, block) %>%
  summarise(RustScore = mean(RustScore))

coffee_na <- 
  coffee %>%
  group_by(site, variety) %>%
  complete(block = 1:3) %>%   # fill RustScore with NA values
  ungroup()

# General boxplot
coffee_na %>% ggplot(aes(x = reorder(site, RustScore), y = RustScore, fill = factor(block))) +
  geom_boxplot() +
  theme_xiaofei()

sites <- unique(coffee_na$site)

# Replace "Kartila 1" by "Kartika 1"
coffee_na <- coffee_na %>%
  mutate(variety = recode(variety, "Kartila1" = "Kartika1")) 

# Abrimos el archivo PDF para guardar los gráficos
#pdf("images/raw_data_rust_site.pdf", width = 10, height = 6)

for (s in sites) {
  cat("Procesando sitio:", s, "\n")
  
  # Filtrar por sitio
  df_site <- coffee_na %>% filter(site == s)
  
  my_plot <- df_site %>%
    ggplot(aes(x = variety, y = RustScore)) +
    geom_boxplot() +
    geom_point(aes(color = factor(block))) +
    theme_xiaofei() +
    labs(title = s)
  
  # Imprimir el gráfico en una nueva página del PDF
  print(my_plot)
}

# Cerramos el archivo PDF
#dev.off()

master_data <- list()

master_data[["raw_data"]] <- coffee_na

var <- "RustScore"
summ_design <- coffee_na %>%
  distinct(site, block, variety, .data[[var]]) %>%
  group_by(site) %>%
  summarise(
    n_gen = n_distinct(variety),
    n_reps = n_distinct(block),
    n_total = n(),
    n_missing = sum(is.na(.data[[var]])),
    n_percent = n_missing / n_total,
    zeros = sum(.data[[var]] == 0, na.rm = TRUE),
    rcbd = ifelse(n_reps > 1, TRUE, FALSE)
  ) %>%
  arrange(n_gen)

summ_design <- summ_design %>%
  as.data.frame() %>%
  mutate(trait = var) 

traits <- "RustScore"
group_var <- "site"
data <- coffee_na

summ_traits <- data %>%
  dplyr::select(all_of(group_var), all_of(traits)) %>%
  pivot_longer(cols = all_of(traits), names_to = "trait", values_to = "value") %>%
  group_by(.data[[group_var]], trait) %>%
  summarise(
    Min = min(value, na.rm = TRUE),
    Mean = mean(value, na.rm = TRUE),
    Median = median(value, na.rm = TRUE),
    Max = max(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    CV = SD / abs(Mean),
    n = n(),
    n_miss = sum(is.na(value)),
    miss_perc = n_miss / n,
    .groups = "drop"
  )

summary_combined <- left_join(summ_traits, summ_design)

master_data[[paste0("summary")]] <- summary_combined

# Almacenar resultados
blue_list <- list()
blup_list <- list()
h2_list <- list()
h2_cullis <- c()
weights_list <- list()

# Lista de sitios
sites <- unique(coffee_na$site)
sites <- sites[!sites %in% c("StAndrew_JAM", "StAnn_JAM")] # because one rep for Jamaica sites


# Dentro del loop, justo después de filtrar df_site
coffee_na <- coffee_na %>%
  mutate(
    variety = as.factor(variety),
    block = as.factor(block)
  )

for (s in sites) {
  cat("Procesando sitio:", s, "\n")
  
  # Filtrar por sitio
  df_site <- coffee_na %>% filter(site == s)
  
  # Modelo para BLUEs (variety como fijo)
  model_blue <- asreml(
    fixed = RustScore ~ block + variety,
    ai.sing = TRUE,
    data = df_site
  )
  
  # Extraer BLUEs
  blue <- predict(model_blue, classify = "variety")$pvals %>%
    select(variety, BLUE = predicted.value, se = std.error) %>%
    mutate(site = s)
  blue_list[[s]] <- blue
  
  # Extraer pesos (inverse of prediction error variance)
  weights <- 1 / (blue$se^2)
  weights_list[[s]] <- tibble(site = s, variety = blue$variety, weight = weights)
  
  
  # Modelo para BLUPs y H² (variety como aleatorio)
  model_blup <- asreml(
    fixed = RustScore ~ block,
    random = ~ variety,
    ai.sing = TRUE,
    data = df_site
  )
  
  model_blup <- update.asreml(model_blup)
  
  # Extraer BLUPs
  blup <- predict(model_blup, classify = "variety")$pvals %>%
    select(variety, BLUP = predicted.value, se = std.error) %>%
    mutate(site = s)
  blup_list[[s]] <- blup
  
  
  # Heredabilidad por sitio: h² = σ²_g / (σ²_g + σ²_e / r) # we might change the calculation of heritability using Cullis
  varcomp <- summary(model_blup)$varcomp
  sigma_g <- varcomp["variety", "component"]
  sigma_e <- varcomp["units!R", "component"]
  n_reps <- df_site %>% count(variety) %>% pull(n) %>% mean()
  h2 <- sigma_g / (sigma_g + sigma_e / n_reps)
  
  # variance due to `gen`
  sg2 <- summary(model_blup)$varcomp[1, 'component']
  # mean variance of a difference of two BLUPs
  vblup <- predict(model_blup , classify ="variety")$avsed ^ 2
  h2.cullis <- 1-(vblup / 2 / sg2)
  h2_cullis[[s]] <- tibble(site = s, h2_Cullis = h2.cullis)
  
  h2_list[[s]] <- tibble(site = s, h2 = h2)
}

# Combinar resultados
blue_all <- bind_rows(blue_list)
weights_all <- bind_rows(weights_list)
# put together BLUEs and weights
blue_all <- blue_all %>% left_join(weights_all)

blup_all <- bind_rows(blup_list)
h2_all <- bind_rows(h2_list)
h2_all_cullis <- bind_rows(h2_cullis)
h2_all <- h2_all %>% left_join(h2_all_cullis)


master_data[[paste0("WCR_","single_BLUEs_weights")]] <- blue_all 
master_data[[paste0("WCR_","single_BLUPs")]] <- blup_all 
master_data[[paste0("WCR_","single_heritability")]] <- h2_all 

# connectivity plot with non sites from Jamaica sites %in% c("StAndrew_JAM", "StAnn_JAM")
connect_mat <- check_connectivity(data = coffee_na %>% filter(!site %in% c("StAndrew_JAM", "StAnn_JAM")),
                                  genotype = "variety",
                                  trial = "site",
                                  response = "RustScore",
                                  all = T, return_matrix = T)

master_data[[paste0("WCR_","connect_mat")]] <- connect_mat #

#pdf(paste("images/connec_rust_fa2", Sys.Date(), sep = "_", ".pdf"), width = 8, height = 6) 

connect_mat %>% as.data.frame() %>% 
  rownames_to_column("site1") %>%
  pivot_longer(-site1, names_to = "site2", values_to = "n_common") %>% 
  ggplot(aes(x = site1, y = site2, fill = n_common)) +
  geom_tile(color = "white") +
  geom_text(aes(label = n_common), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal(base_size = 12) +
  theme_xiaofei() +
  labs(x = "site", y = "site")

#dev.off()

#ggsave(paste("images\\connectivity_plot_RustScore", ".png"),
#       units = "in", dpi = 300, width = 10, height = 8)

# remove from the single analysis Buginyanya_UGA and Mzuzu_MWI (low heritability)
# blues_full <- blue_all %>% filter(!site %in% c("Mzuzu_MWI", "Buginyanya_UGA"))
# Salvador comment: 
# Although Mzuzu_MWI and Buginyanya_UGA showed low single-environment heritability 
# for rust severity, they were retained in the multi-environment analysis because 
# both environments were well connected to the remaining trial. Given the ordinal 
# nature and low prevalence of rust symptoms in several sites, low heritability was 
# interpreted cautiously and not used as the sole exclusion criterion for rust.

blues_full <- blue_all
blues_full$site <- as.factor(blues_full$site)


# Ajustar modelo multiambiental con FA y estrutura de error sencilla puesto ~ units
# BLUEs ya estaban ponderados (con weights), y las varianzas específicas por sitio 
# estaban parcialmente absorbidas por los errores del modelo univariado
# factor analystics 2

mta_fa_model <- asreml(
  fixed = BLUE ~ site,
  random = ~ fa(site, 2):variety,
  residual = ~ units,
  weights = weight,
  data = blues_full,
  ai.sing = TRUE,
  maxit = 100
)

mta_fa_model <- update.asreml(mta_fa_model)
mta_fa_model <- update.asreml(mta_fa_model)
mta_fa_model <- update.asreml(mta_fa_model)

# double check 
#pdf(paste("images/plot_assumption_rust", Sys.Date(), sep = "_", ".pdf"), width = 6, height = 6) 
plot(mta_fa_model)
#dev.off()

# BLUPs por sitio y variedad (incluyen interacción)
blup_fa <- predict(mta_fa_model, classify = "variety:site")$pvals
head(blup_fa)
master_data[[paste0("WCR_","GxE_BLUPs_site_variety")]] <- blup_fa 


# prediction across sites
predsALL <- predict(mta_fa_model, classify = "variety")$pvals
head(predsALL)
master_data[[paste0("WCR_","GxE_BLUPs_variety")]] <- predsALL 

# Standard errors 
# BLUPs - SE 
blups_se_gen <- predict(mta_fa_model, classify = "variety")$pvals %>%
  mutate(
    CI_lower = predicted.value - 1.96 * std.error,
    CI_upper = predicted.value + 1.96 * std.error
  )

# stability  using variance - Less, more stable
blups_site_var <- predict(mta_fa_model, classify = "variety:site")$pvals %>%
  group_by(variety) %>%
  summarise(
    stability = var(predicted.value, na.rm = T), # 
    mean_blup = mean(predicted.value, na.rm = T)
  )

comments_review <- list()
# save CI and std.error
comments_review[["standar_error_yield"]]<- blups_se_gen |> select(-status)

# save stability across enviroments
comments_review[["stability_index_yield"]]<- blups_site_var

# save excel file
#folder_output <- here::here("output//")
#meta_file_name <- paste0(folder_output, paste("comments", "rust", "master_results", Sys.Date(), ".xlsx", sep = "_"))
#write.xlsx(comments_review, file = meta_file_name)



# quality parameters
aic_2 <- summary(mta_fa_model)$aic
bic_2 <- summary(mta_fa_model)$bic
mta_fa_model$loglik

# parameters to estimate
# Site numbers
s <- 24
k <- 2 # 2 factor
n_parameters_2 <- s * (k + 1) - (k * (k - 1)) / 2

# extract variance components
vars_2 <- lucid::vc(mta_fa_model)

# extract loadings or factors
fa1.loadings =  vars_2[grepl("!fa1$", vars_2$effect), "component"]
fa2.loadings =  vars_2[grepl("!fa2$", vars_2$effect), "component"]
#fa3.loadings =  vars_2[grepl("!fa3$", vars$effect), "component"]

# Loading of the mta_fa_model 
L = as.matrix(cbind(fa1.loadings, fa2.loadings))

# double check orthogonality - covariance matrix between factors based on the 
# environmental loadings of the model - out of diagonal should be close to zero
crossprod(L)  #t(L)%*%L cross product -> covariance matrix
# The Loading in ASReml-R are not orthogonal, so it’s neccesary to perform an orthogonal rotation for these solutions.

# Applying the singular value descomposition to the loadings
svd.L = svd(L)     #  L = UDV'
str(svd.L)

L_reconstructed <- svd.L$u%*%diag(svd.L$d)%*%t(svd.L$v)

max(abs(L - L_reconstructed)) # 8.881784e-16
all.equal(L, L_reconstructed, tolerance = 1e-10)

# Rotation. This consist in multiply the loadings by the V matrix for the right:
L.star = L%*%svd.L$v 

#  there are −1′s pre-multiplying the loadings
L.star <- L.star*-1

#  environment-specific genetic variances Ψ
psi = vars_2[grepl("!var$", vars_2$effect), "component"]

# Variance-covariance matrix
# The elements of the diagonal are the genotypic variances in each environment, 
# and the non-diagonal elements are the pairs of genotypic covariances between environments.
Gvar <- L.star %*% t(L.star) + diag(psi)

# Variance explained
VarTot = sum(diag(L.star %*% t(L.star))) / sum(diag(L.star %*% t(L.star) + diag(psi) ))
VarTot


# Número de sitios
s <- length(unique(blues_full$site))

# Lista para almacenar resultados
results <- list()

# Loop sobre k = 1 a 4
for (k in 1:4) {
  cat("Fitting model FA", k, "\n")

  # Ajustar modelo FA(k)
  mta_fa_model <- asreml(
    fixed = BLUE ~ site,
    random = ~ fa(site, k):variety,
    residual = ~ units,
    weights = weight,
    data = blues_full,
    ai.sing = TRUE
  )

  # Actualizar modelo hasta convergencia
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)
  mta_fa_model <- update.asreml(mta_fa_model)

  # Extraer criterios de información
  aic <- summary(mta_fa_model)$aic
  bic <- summary(mta_fa_model)$bic
  loglik <- mta_fa_model$loglik

  # Calcular número de parámetros estimados
  n_parameters <- s * (k + 1) - (k * (k - 1)) / 2

  # Extraer componentes de varianza
  vars <- lucid::vc(mta_fa_model)

  # Extraer loadings
  loading_pattern <- paste0("!fa", 1:k, "$")

  # Buscar índices de cada loading
  loading_indices <- lapply(loading_pattern, function(pattern) {
    grep(pattern, vars$effect)
  })

  # Extraer componentes y combinarlos en matriz
  fa_loadings <- do.call(cbind, lapply(loading_indices, function(i) vars$component[i]))

  # Asignar nombres a columnas
  colnames(fa_loadings) <- paste0("fa", 1:k)


  # Construir matriz de loadings
  L <- as.matrix(fa_loadings)

  # Rotación ortogonal usando SVD
  svd_L <- svd(L)
  L_star <- L %*% svd_L$v * -1  # Multiplicamos por -1 para mantener consistencia de signos

  # Extraer varianzas específicas por ambiente (psi)
  psi_indices <- grep("!var$", vars$effect)
  psi <- vars$component[psi_indices]

  # Calcular matriz de varianzas genotípicas por ambiente
  Gvar <- L_star %*% t(L_star) + diag(psi)

  # Calcular varianza explicada
  VarTot <- sum(diag(L_star %*% t(L_star))) / sum(diag(Gvar))

  # Almacenar resultados
  results[[paste0("FA", k)]] <- data.frame(
    Model = paste0("FA", k),
    AIC = aic,
    BIC = bic,
    LogL = loglik,
    Parameters = n_parameters,
    VE = round(VarTot * 100, 2)
  )
}

# Model parameters (Fa1 vs Fa2)
results_df <- bind_rows(results)
results_df 

#  get the variance explained for each experiment
ns <- nlevels(as.factor(blup_fa$site))
k <- 2 
snam <- levels(as.factor(blup_fa$site))
paf.site <- matrix(0, nrow = ns, ncol = k)
dimnames(paf.site) <- list(snam, paste("fac", 1:k, sep = "_"))
for (i in 1:k) {
  paf.site[, i] <- 100 * L.star[, i]^2 / (rowSums(L.star^2) + psi)
}

if (k > 1) {
  all <- 100 * diag(L.star %*% t(L.star))/
    diag(L.star %*% t(L.star) + diag(psi) )
  paf.site <- cbind(paf.site, all)
}


# calculate the percentage of genetic variance accounted by FA1 and FA2?
VarTot <- VarTot
VarGenEnv <- diag(L.star %*% t(L.star) + diag(psi) )
TotGenVar <- sum(VarGenEnv)
TotGenVar

VarFA1 <- sum(VarGenEnv*paf.site[,1])/100
VarFA2 <- sum(VarGenEnv*paf.site[,2])/100

PerVarFA1 <- VarFA1/TotGenVar
PerVarFA2 <- VarFA2/TotGenVar

c(PerVarFA1, PerVarFA2)


# matrix de correlaciones entre los sitios
Cmat <- cov2cor(Gvar) 
rownames(Cmat) <- snam
colnames(Cmat) <- snam

master_data[[paste0("WCR_","geno_cor")]] <- Cmat 
Cmat 

# heatmap
heatmap(Cmat,
        scale = "none",
        col = heat.colors(100),
        margins = c(10, 10))  # deja espacio para que los nombres se vean

#install.packages("pheatmap")
library(pheatmap)
#png(filename = "images/Rust_Score_Fa2/corr_sites_fa2_Rust.png",
#    width = 12, height = 6, units = "in", res = 300)

pheatmap(Cmat, 
         cluster_rows = TRUE, 
         cluster_cols = TRUE,
         display_numbers = TRUE, 
         color = colorRampPalette(c("white", "orange", "red"))(100))

#dev.off()

# Vector PDF output
library(corrplot)
# pdf(file = "images/Rust_Score_Fa2/corr_sites_fa2_Rust.pdf",
#     width = 8, height = 8,
#     useDingbats = FALSE,   # avoids dingbat fonts; better portability
#     family = "sans")       # optional: choose a common font

corrplot(Cmat, order = "hclust", type = "lower", diag = FALSE, tl.cex = 0.7)

#dev.off()


# Extraer todos los efectos fijos del modelo
cof <- coef(mta_fa_model)$fixed %>%
  as.data.frame() %>%
  rownames_to_column("coef")

# Separar el intercepto
interc <- cof %>%
  filter(coef == "(Intercept)") %>%
  pull(effect)

# Extraer los efectos fijos por sitio
env_means <- cof %>%
  filter(str_detect(coef, "^site_")) %>%
  mutate(site = str_remove(coef, "^site_")) %>%
  rename(coef_effect = effect)

# Calcular BLUEs sumando intercepto + efecto de ambiente
env_means <- env_means %>%
  mutate(BLUE = interc + coef_effect)

# Resultado final:
print(env_means)

faComp <- data.frame(site = levels(blup_fa$site),    # Environments names
                     fa1  = L.star[,1],                # Comp 1
                     fa2  = L.star[,2],                # Comp 2
                     psi  = psi,                       # Specific genetic variances
                     Vg   = diag(Gvar),                # Genotypic Variances
                     BLUE = env_means$BLUE )           # BLUE by Environmet  


master_data[["fa1_fa2_sites"]] <- faComp

# Visualization
library(ggrepel)
d=data.frame(x=rep(0, nrow(L.star)), y=rep(0, nrow(L.star)), vx=L.star[,1], vy=L.star[,2])
loadings = ggplot(faComp, aes(x = fa1, y = fa2)) + 
  geom_point(aes(colour = Vg, size = BLUE)) +
  scale_colour_gradient(low = "pink", high = "blue") + 
  geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1, label.size = 0.05) + 
  ggtitle("Environment Factor Loadings") +
  xlab("FA1 loading") + ylab("FA2 loading") + theme_bw(base_size = 15)+
  geom_vline(xintercept = 0,linetype = 2) + geom_hline(yintercept = 0,linetype = 2) +
  theme_xiaofei()

loadings
# ggsave(paste("images\\loadings_Rust_Score", ".png"),
#        units = "in", dpi = 300, width = 12, height = 6)

mod_fa1 <- lm(BLUE ~ fa1, data = faComp)
summary(mod_fa1)
p1 <- ggplot(faComp, aes(x = fa1, y = BLUE)) +
  geom_point(aes(color = site), size = 3, show.legend = F) +
  geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1, label.size = 0.05) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  annotate("text", x = Inf, y = Inf, hjust = 1.2, vjust = 1.5,
           label = paste0("R² = ", round(summary(mod_fa1)$adj.r.squared, 2),
                          "\nP = ", signif(summary(mod_fa1)$coefficients[2, 4], 2))) +
  labs(
    x = "First factor loadings (FA1)",
    y = "Predicted site means (BLUE)"
  ) +
  theme_xiaofei() 

mod_fa2 <- lm(BLUE ~ fa2, data = faComp)
p2 <- ggplot(faComp, aes(x = fa2, y = BLUE)) +
  geom_point(aes(color = site), size = 3, show.legend = F) +
  geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1, label.size = 0.05) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  annotate("text", x = Inf, y = Inf, hjust = 1.2, vjust = 1.5,
           label = paste0("R² = ", round(summary(mod_fa2)$r.squared, 2),
                          "\nP = ", signif(summary(mod_fa2)$coefficients[2, 4], 2))) +
  labs(
    x = "First factor loadings (FA2)",
    y = "Predicted site means (BLUE)"
  ) +
  theme_xiaofei() 
library(patchwork)
p1 + p2 + plot_layout(guides = "collect")
# ggsave(paste("images\\latent_reg_plot_sites_Rust_Score", ".png"),
#        units = "in", dpi = 300, width = 14, height = 6)

# biplot
fa1_scores = coef(mta_fa_model)$random[grep("Comp1:variety", rownames(coef(mta_fa_model)$random)),]
fa2_scores = coef(mta_fa_model)$random[grep("Comp2:variety", rownames(coef(mta_fa_model)$random)),]

names(fa1_scores) = sub("fa(site, 2)_Comp1:variety_", "",names(fa1_scores), fixed = T)
names(fa2_scores) = sub("fa(site, 2)_Comp2:variety_", "",names(fa2_scores), fixed = T)

f = rbind(as.matrix(fa1_scores), as.matrix(fa2_scores))

# We need to rotate the varieties scores as well
nGenotype <- nlevels(blup_fa$variety)                          # 64
f.star = kronecker(t(svd.L$v), diag(nGenotype))%*%f*-1        # f* = (Vt*I)f
rownames(f.star) <- rownames(f)

fa12_scores = merge(f.star[1:nGenotype,1], 
                    f.star[(nGenotype+1):(2*nGenotype),1], by = "row.names")
names(fa12_scores) = c("Genotype", "fa1", "fa2")

#fa12_scores$Score <-  ifelse(sqrt(fa12_scores$fa1^2+fa12_scores$fa2^2)>1.2,1,0)

# explained variance per each accession
fa12_scores <- fa12_scores %>% mutate(fa1_sq = fa1^2, 
                                      fa2_sq = fa2^2,
                                      total = fa1_sq + fa2_sq,
                                      fa1_pct = 100 * fa1_sq / total,
                                      fa2_pct = 100 * fa2_sq / total,
                                      unexplained = 100 - fa1_pct - fa2_pct) %>% 
  mutate(across(where(is.numeric), ~ round(.x, 2)))

master_data[["varieties_scores_fa"]] <- fa12_scores

biplot = ggplot(faComp, aes(x = fa1, y = fa2)) + 
  geom_point() + #aes(colour = Vg, size = BLUE)) +
  #scale_colour_gradient(low = "pink", high = "blue") + 
  geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1) + 
  ggtitle("Environment Factor Loadings") +
  xlab("FA1 loading") + ylab("FA2 loading") + theme_xiaofei()+
  geom_vline(xintercept = 0,linetype = 2) + geom_hline(yintercept = 0,linetype = 2)+
  geom_segment(data=d,
               mapping=aes(x=x, y=y, xend=x+vx, yend=y+vy),
               arrow=arrow(), size=0.5, color="black", alpha=0.3) 
biplot +
  geom_label_repel(data = fa12_scores, aes(label = Genotype),
                   colour = "red",segment.colour = "red" , size=2, alpha=1)  
# ggsave(paste("images\\latent_reg_biplot_Rust_Score", ".png"),
#        units = "in", dpi = 300, width = 12, height = 8)

faCompR <- faComp

faCompR[,2:3] <- diag(1/sqrt(diag(Gvar))) %*% L.star 

d <- data.frame(x=rep(0, nrow(L.star)), y=rep(0, nrow(L.star)), 
                vx=faCompR[,2], vy=faCompR[,3]) 

circleFun = function(center = c(0,0),diameter = 1, npoints = 100){
  r = diameter / 2
  tt <- seq(0,2*pi,length.out = npoints)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}


circle <- circleFun(c(0,0),2,npoints = 100)
ggplot(faCompR, aes(x = fa1, y = fa2)) + 
  geom_point(aes(colour = Vg, size = BLUE)) +
  scale_colour_gradient(low = "pink", high = "blue") + 
  geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1, label.size = 0.01) + 
  ggtitle("Environment Factor Loadings") +
  xlab("FA1 loading") + ylab("FA2 loading") + theme_xiaofei()+
  geom_vline(xintercept = 0,linetype = 2) + geom_hline(yintercept = 0,linetype = 2)+
  geom_segment(data=d,
               mapping=aes(x=x, y=y, xend=x+vx, yend=y+vy),
               arrow=arrow(), size=0.5, color="black", alpha=0.3) +
  geom_path(data=circle, aes(x,y))
# ggsave(paste("images\\circle_cor_loadings_Rust_Score", ".png"),
#        units = "in", dpi = 300, width = 9, height = 8)

# Residual variance and H2
VarE <- lucid::vc(mta_fa_model)[grep(lucid::vc(mta_fa_model)$effect,pattern = "!R"),] %>% 
  mutate_if(is.numeric, round, digits=2)  %>% 
  mutate(effect= gsub(pattern = "!R",x = effect,replacement = "") )

# VarE = 1.96
trG <- sum(diag(Gvar))   # Suma de varianzas genéticas por sitio
s <- nrow(Gvar)          # Número de sitios
(h2 <- trG / (trG + s * 0.87))
h2

# cullis heritability
# 1. Obtener los BLUPs y errores estándar
blups <- predict(mta_fa_model, classify = "variety")$pvals

# 2. Calcular PEV (error estándar al cuadrado)
blups <- blups %>%
  filter(status == "Estimable") %>%
  mutate(PEV = std.error^2)

# 3. Calcular la varianza genética promedio (opcional: usar promedio de var() de cada sitio FA si querés por sitio)
var_gen <- mean(lucid::vc(mta_fa_model)$component[grep("!var", lucid::vc(mta_fa_model)$effect)])

# 4. Calcular la heredabilidad tipo Cullis
H2_Cullis <- 1 - mean(blups$PEV) / (2 * var_gen)

# 5. Resultado
H2_Cullis 

# Genetic prediction
bv <- summary(mta_fa_model,coef=TRUE)$coef.random  
alln<-row.names(bv)
aimn<-alln[grep('fa\\(.*,.*\\)',alln)] # extrae elementos de GxE
Xfasln<-bv[aimn,]
Xfasln_1 <- Xfasln[,1] 

Xfa2 <- matrix(Xfasln_1,nrow=nlevels(blup_fa$variety), dimnames = list(levels(blup_fa$variety)))
Xfa2 <- data.frame(Xfa2)
names(Xfa2) <- c(levels(blup_fa$site), "fa1", "fa2")
Xfa2 <- Xfa2 %>% rownames_to_column(., "Genotype") %>% .[,1:21]
head(Xfa2)

Xfa2_blup <- Xfa2 %>% pivot_longer(!Genotype, names_to = "site", values_to = "blup" )


# ---- Predicitions ----
predictions <- Xfa2_blup %>%
  left_join(env_means, by = "site")  %>%  # Une BLUE por sitio
  mutate(predicted_value = BLUE + blup)

# example with any accession
Xfa2[Xfa2$Genotype=="AB3",]

Ug <- kronecker( L.star , diag( nlevels(blup_fa$variety)) ) %*% (f.star)  
Ug <- matrix(Ug,ncol = nlevels(blup_fa$site), 
             dimnames = list(levels(blup_fa$variety),levels(blup_fa$site)))

Ug <- data.frame(Ug) %>% rownames_to_column("Genotype")
head(Ug)

# FAST
#Xfa2 is the genotypic by environment prediction incorporing the site-specific variance, and Ug is only based on factors.

Xfa2 <- Xfa2 %>% gather(data = ., key = "site", value = "blup", -1 )
Ug <- Ug %>% gather(data = ., key = "site", value = "regblup", -1 )

UgOve <- merge(Xfa2, Ug, by = c("Genotype", "site"))
head(UgOve)

FAST <- merge(UgOve, faComp[,1:3], by = "site")  
head(FAST)


accessions <- unique(FAST$Genotype)

# Abrimos el archivo PDF para guardar los gráficos
#pdf("images/latent_plot_by_acc.pdf", width = 10, height = 6)

for (acc in accessions) {
  cat("Proccessing variety:", acc, "\n")
  
  
  p <- FAST %>% 
    filter(Genotype == acc) %>% 
    gather(., key = "loading", value = "value", -1,-2,-3,-4) %>% 
    ggplot(aes(x=value,y=blup))+
    geom_point(size=3)+
    facet_grid(Genotype~loading, scales = "free_x")+
    theme_bw(base_size = 15)+
    geom_smooth(aes(x=value,y=blup), method = "lm", formula = y~x, se = F)+
    ggpubr::stat_regline_equation(
      aes(x = value, y = blup, label = paste(..rr.label.., sep = "~~~")),
      size = 5,
      label.x.npc = 0.5,
      label.y.npc = 1
    ) +
    #ggpubr::stat_regline_equation(aes(x=value,y=regblup), size=5)+
    geom_hline(yintercept = 0,linetype = 2,color="grey")+
    geom_label_repel(aes(label = site), nudge_y= 0.05, nudge_x=-0.03, force=1)
  
  print(p) 
  
}

# Cerramos el archivo PDF
#dev.off()

# facets by yield latent regression plot
#saveRDS(FAST, "FAST_rust.rds")
#FAST <- readRDS("FAST_rust.rds")

# remove from FAST BP429A
FAST <- FAST %>% filter(!Genotype ==  "BP429A")

#pdf("images/Fa2/latent_plots_rust.pdf", width = 12, height = 8)

# Fa1
FAST %>% 
  gather(., key = "loading", value = "value", -1,-2,-3,-4) %>% 
  filter(loading == "fa1") %>% 
  ggplot(aes(x=value,y=blup))+
  facet_wrap(~Genotype, scales = "free_x", ncol = 8) +
  geom_point(aes(color = site))+
  geom_smooth(method = "lm", se = F) +
  theme_bw(base_size = 15) +
  ggpubr::stat_regline_equation(
    aes(x = value, y = blup, label = paste(..rr.label.., sep = "~~~")),
    size = 3,
    label.x.npc = 0.1,
    label.y.npc = 1
  )+
  theme(strip.text = element_text(size = 8),
        axis.text.x = element_text(size = 8),  # X-axis tick labels
        axis.text.y = element_text(size = 8)) +
  labs(x = "First factor loading")

FAST %>% 
  gather(., key = "loading", value = "value", -1,-2,-3,-4) %>% 
  filter(loading == "fa2") %>% 
  ggplot(aes(x=value,y=blup))+
  facet_wrap(~Genotype, scales = "free_x", ncol = 8) +
  geom_point(aes(color = site))+
  geom_smooth(method = "lm", se = F) +
  theme_bw(base_size = 15) +
  ggpubr::stat_regline_equation(
    aes(x = value, y = blup, label = paste(..rr.label.., sep = "~~~")),
    size = 3,
    label.x.npc = 0.1,
    label.y.npc = 1
  )+
  theme(strip.text = element_text(size = 8),
        axis.text.x = element_text(size = 8),  # X-axis tick labels
        axis.text.y = element_text(size = 8)) +
  labs(x = "Second factor loading")

#dev.off()

master_data[["scores_factors_regblup"]] <- FAST

# save excel file
#folder_output <- here::here("output//")
#meta_file_name <- paste0(folder_output, paste("2025", "WCR", "master_results_Rust_Score", Sys.Date(), ".xlsx", sep = "_"))
#write.xlsx(master_data, file = meta_file_name)
# Other way to plot latent plot

accessions <- unique(FAST$Genotype)

#pdf("images/latent_plot_by_acc.pdf", width = 10, height = 6)

for (acc in accessions) {
  cat("Procesando variedad:", acc, "\n")
  
  df_acc <- FAST %>%
    filter(Genotype == acc) %>%
    gather(key = "loading", value = "value", -Genotype, -site, -blup, -regblup)
  
  # Calcular R2 por loading para este Genotype
  r2_text <- df_acc %>%
    group_by(loading) %>%
    summarise(
      r2 = summary(lm(blup ~ value))$r.squared,
      x = max(value, na.rm = TRUE),
      y = max(blup, na.rm = TRUE)
    )
  
  p <- df_acc %>%
    ggplot(aes(x = value, y = blup)) +
    geom_point(size = 3) +
    facet_grid(Genotype ~ loading, scales = "free_x") +
    theme_bw(base_size = 15) +
    geom_smooth(aes(y = regblup), method = "lm", formula = y ~ x, se = FALSE, color = "blue") +
    geom_label_repel(aes(label = site), nudge_y = 0.05, nudge_x = -0.03, force = 1) +
    geom_hline(yintercept = 0, linetype = 2, color = "grey") +
    geom_text(data = r2_text,
              aes(x = x, y = y, label = paste0("R² = ", round(r2, 2))),
              inherit.aes = FALSE,
              hjust = 1, vjust = 1, size = 4)
  
  print(p)
}

#dev.off()



