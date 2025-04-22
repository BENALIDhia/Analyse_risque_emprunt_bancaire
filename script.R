#vider la mémoire
rm(list = ls())
library(corrplot)
library(gtsummary)
library(GGally)
library(forestmodel)
library(effects)
library(report)
library(kernlab)
library(MASS) 
library(randomForest) 
library(rpart)
library (e1071)
library(class)
library(ROCR) 

#### Modèle de régression logistique binomial pour la classification binaire ####

bankloans <- read.csv("C:/Users/MSI/Desktop/M2 Intelligence Artificielle/Apprentissage stat et data mining/projet/archive (1)/bankloans.csv", stringsAsFactors=TRUE)
View(bankloans)
str(bankloans)

################################################################################
### prétraitement des données###
################################################################################

#supprimer les ligne ou il'ya des valeurs manquantes
bankloans <- na.omit(bankloans)
#str(bankloans)

#la variable ed est une variable categoriale mais elle est importé comme une variable numerique
#On la change en variable categorielle en utilisant la commande as.factors
bankloans$ed<-as.factor(bankloans$ed)
str(bankloans)

################################################################################
### Modèle complet###
################################################################################

# matrcie de correlation des variables
X <- model.matrix(default ~., data = bankloans)[,-1]
XX <- cbind(as.data.frame(X), default = bankloans[,"default"])
# Calcul de la matrice de corrélation
correlation_matrix <- cor(XX)
corrplot(correlation_matrix, method = "circle")
# Affichage de la matrice de corrélation
print(correlation_matrix)


#faire une régression logistique de la variable binaire default en fonction des variables (explicatives) 
#de la bd bankloans :
modele.RL <- glm(formula = default~ .,  data = bankloans, family = binomial)


#Affichage
print(modele.RL)
summary(modele.RL)
attributes(modele.RL)
tbl_regression(modele.RL, exponentiate = FALSE)
ggcoef_model(modele.RL, exponentiate = FALSE)
forest_model(modele.RL, exponentiate = FALSE)
plot(allEffects(modele.RL))
report(modele.RL)

#### Tester (avec rapport de vraisemblance) la validité du modèle complet ####

# i.e., tester  H0 : ``w1=0, ..., wp+1=0'' contre H1 : ``le contraire de H_0'' 
Sn <- modele.RL$null.deviance - modele.RL$deviance #la statistique du rapport de vraisemblance
print(Sn)
ddl <- modele.RL$df.null - modele.RL$df.residual #nombre de degrés de liberté de la loi limite de Sn, sous H_0
print(ddl)
pvalue <- pchisq(q = Sn, df = ddl, lower.tail = F) #p_value du test : P(Z>Sn) où Z suit une loi du chi^2(ddl)
print(pvalue) #on obtient 1.253064e-27, on rejette H0, donc le modèle est "très" significatif





################################################################################
#### Sélection de modèles (de variables) selon les critères AIC, AICC et BIC ###
################################################################################

#### Recherche exhaustive ####
library(glmulti)
#AIC
select.modele.aic <- glmulti(default~.,  data = bankloans, family = binomial, level = 1, 
                             fitfunction = glm, crit = "aic", 
                             plotty = FALSE, method = "h")
#BIC
select.modele.bic <- glmulti(default ~.,  data = bankloans, family = binomial, level = 1, 
                             fitfunction = glm, crit = "bic", 
                             plotty = FALSE, method = "h")
#AICC
select.modele.aicc <- glmulti(default ~.,  data = bankloans, family = binomial, level = 1, 
                              fitfunction = glm, crit = "aicc", 
                              plotty = FALSE, method = "h")

summary(select.modele.aic)$bestmodel
summary(select.modele.aicc)$bestmodel
summary(select.modele.bic)$bestmodel

##Si on veut choisir parmi les variables de la matrice de design, on fait comme suit :

#supprimer les ligne ou il'ya des valeurs manquantes
bankloans <- na.omit(bankloans)
XX <- model.matrix(default ~., data = bankloans)[,-1] #cette fonction construit la matrice de design en remplaçant 
#chacune des variables qualitatives pour les indicatrices 
#de ses modalités (la première modalité est supprimée)
#on supprime la première colonne correspondant à l'intercept
bankloans.num.data <- cbind(as.data.frame(XX), default = as.factor(bankloans[,"default"])) #bd constituée que de variables explicatives numériques 
#et une variable réponse qualitative (binaire) 
str(bankloans.num.data)
View(bankloans.num.data)

#AIC
select.modele.num.aic <- glmulti(default ~.,  data = bankloans.num.data, family = binomial, level = 1, 
                                 fitfunction = glm, crit = "aic", 
                                 plotty = FALSE, method = "h")
#BIC
select.modele.num.bic <- glmulti(default ~.,  data = bankloans.num.data, family = binomial, level = 1, 
                                 fitfunction = glm, crit = "bic", 
                                 plotty = FALSE, method = "h")
#AICC
select.modele.num.aicc <- glmulti(default ~.,  data = bankloans.num.data, family = binomial, level = 1, 
                                  fitfunction = glm, crit = "aicc", 
                                  plotty = FALSE, method = "h")

summary(select.modele.num.aic)$bestmodel
summary(select.modele.num.aicc)$bestmodel
summary(select.modele.num.bic)$bestmodel




################################################################################
#### Estimation de l'erreur de classification par les méthodes de validation croisée ####
################################################################################

# Nous allons évaluer l'erreur de classification issue du modèle optimal selon le critère BIC
str(bankloans.num.data)

#rappelons le modèle optimal selon BIC
modele.opt.bic.formula <- summary(select.modele.num.bic)$bestmodel
modele.opt.bic.formula

n <- nrow(bankloans.num.data)

library(boot)
bankloans_modele_glm <- glm(formula = modele.opt.bic.formula, data = bankloans.num.data, family = binomial)
cout <- function(r, pi) mean(abs(r-pi) > 0.5) 
#le cas K = n, n étant le nombre d'observations, correspond à la méthode leave-one-out :
n <- nrow(bankloans.num.data)
K <- n
cv.err <- cv.glm(data = bankloans.num.data, glmfit = bankloans_modele_glm, cost = cout, K = K)
cv.err$delta[1] 

#comparaison avec le modèle complet
modele.glm.complet <- glm(formula = default ~., data = bankloans.num.data, family = binomial)
cv.err.modele.complet <- cv.glm(data = bankloans.num.data, 
                                glmfit = modele.glm.complet, cost = cout, K = K)
cv.err.modele.complet$delta[1] 




################################################################################
###matrice de confusion###
################################################################################

## modèle global
# Ajuster le modèle global
bankloans_modele_glm <- glm(formula = default ~., data = bankloans.num.data, family = binomial)

# Obtenez les prédictions du modèle global
pi_mod_glob <- predict(bankloans_modele_glm, type = "response")

# Définissez la matrice de confusion pour le modèle global
conf_matrix_mod_glob <- table(reference = bankloans.num.data$default, prediction = ifelse(pi_mod_glob  > 0.5, 1, 0))

# Affichez la matrice de confusion
conf_matrix_mod_glob

# Calcul des résultats (faux positif, vrai positif, faux négatif, vrai négatif)
fp_mod_glob <- conf_matrix_mod_glob[1, 2]
fn_mod_glob <- conf_matrix_mod_glob[2, 1]
vp_mod_glob <- conf_matrix_mod_glob[2, 2]
vn_mod_glob <- conf_matrix_mod_glob[1, 1]

# Affichez les résultats
cat("Faux Positifs (FP) :", fp_mod_glob, "\n")
cat("Faux Négatifs (FN) :", fn_mod_glob, "\n")
cat("Vrais Positifs (VP) :", vp_mod_glob, "\n")
cat("Vrais Négatifs (VN) :", vn_mod_glob, "\n")



## modèle réduit

# Ajuster le modèle optimal selon BIC

bankloans_modele_glm <- glm(formula = modele.opt.bic.formula, data = bankloans.num.data, family = binomial)

# Obtenez les prédictions du modèle optimal selon BIC
pi_opt_bic <- predict(bankloans_modele_glm, type = "response")

# Définissez la matrice de confusion pour le modèle optimal selon BIC
conf_matrix_opt_bic <- table(reference = bankloans.num.data$default, prediction = ifelse(pi_opt_bic > 0.5, 1, 0))

# Affichez la matrice de confusion
conf_matrix_opt_bic

# Calcul des résultats (faux positif, vrai positif, faux négatif, vrai négatif)
fp_opt_bic <- conf_matrix_opt_bic[1, 2]
fn_opt_bic <- conf_matrix_opt_bic[2, 1]
vp_opt_bic <- conf_matrix_opt_bic[2, 2]
vn_opt_bic <- conf_matrix_opt_bic[1, 1]

# Affichez les résultats
cat("Faux Positifs (FP) :", fp_opt_bic, "\n")
cat("Faux Négatifs (FN) :", fn_opt_bic, "\n")
cat("Vrais Positifs (VP) :", vp_opt_bic, "\n")
cat("Vrais Négatifs (VN) :", vn_opt_bic, "\n")

################################################################################
#### classer les variables d'un modèle selon leur niveau de significativité ####
################################################################################


# on utilise une approche test : (utiliser les p_values des tests correspondants, 
# test de Wald et/ou test du rapport de vraisemblance).

# Considérons par exemple le modèle optimal selon BIC 
# et classons les variables par ordre décroissant des valeurs des p_values de chacun des tests de Wald 

#rappelons le modèle optimal selon BIC
modele.opt.bic.formula <- summary(select.modele.num.bic)$bestmodel
modele.opt.bic.formula #ce modèle utilise les v.a. explicatives :"employ + address + debtinc + creddebt"
#la bd correspondante est la suivante
bankloans.opt.data <- bankloans.num.data[,c("employ", "address", "debtinc", "creddebt", 
                                            "default")]
str(bankloans.opt.data)
View(bankloans.opt.data)

modele <- glm(formula = default~., data = bankloans.opt.data, family = binomial)
#Affichage
tbl_regression(modele, exponentiate = FALSE)
ggcoef_model(modele, exponentiate = FALSE)
forest_model(modele, exponentiate = FALSE)
plot(allEffects(modele))

print(modele)
summary(modele)
tab.modele <- summary(modele)$coefficients
tab.modele <- as.data.frame(tab.modele)
str(tab.modele)
View(tab.modele)
vect.des.pvalues.Wald <- tab.modele[,"Pr(>|z|)"]
names(vect.des.pvalues.Wald) <- row.names(tab.modele)

#on supprime la pvalue de l'intercept
vect.des.pvalues.Wald <- vect.des.pvalues.Wald[!(names(vect.des.pvalues.Wald) == "(Intercept)")]
vect.des.pvalues.Wald

#ranger les variables par ordre croissant des valeurs des p_values 
sort(vect.des.pvalues.Wald) # la variable la plus significative est depression, ensuite sexe, ensuite typedouleurD, ensuite tauxmax, ... 

#On classe maintenant les variables par ordre décroissant des valeurs des pvalues de chacun des tests par maximum de vraisemblance 
#### classement des variables par pvalues du test du rapport de vraisemblance, préférable à celui de Wald ####
modele <- glm(formula = default ~ ., data = bankloans.opt.data, family = binomial)
print(modele)

##on procède maintenant au test d'hypothèses par la statistique du rapport de vraisemblance, et au calcul des p_values correspondantes

##Tester l'hypothèse : la variable employ n'est pas significative 
#i.e., tester H_0 : w1 = 0 contre H_1 : w1 != 0  
modele.reduit <- glm(default ~ ., data = bankloans.opt.data[,!(colnames(bankloans.opt.data) == "employ")], family = binomial)
#Statistique du rapport de vraisemblance
Sn <- modele.reduit$deviance - modele$deviance
print(Sn)
pvalue.employ <- pchisq(q = Sn, df = 1, lower.tail = F) #donne P(Z>Sn) où Z est une variable suivant une chi2(1).
print(pvalue.employ)

##Tester H_0 : la variable address n'est pas significative
modele.reduit <- glm(default ~ ., data = bankloans.opt.data[,!(colnames(bankloans.opt.data) 
                                                               == "address")], family = binomial)
#Statistique du rapport de vraisemblance
Sn <- modele.reduit$deviance - modele$deviance
print(Sn)
pvalue.address = pchisq(q = Sn, df = 1, lower.tail = F)
print(pvalue.address)

##Tester H_0 : la variable debtinc n'est pas significative
modele.reduit <- glm(default ~ ., data = bankloans.opt.data[,!(colnames(bankloans.opt.data) 
                                                               == "debtinc")], family = binomial)
#Statistique du rapport de vraisemblance
Sn <- modele.reduit$deviance - modele$deviance
print(Sn)
pvalue.debtinc = pchisq(q = Sn, df= 1, lower.tail = F)
print(pvalue.debtinc)

##Tester H_0 : la variable creddebt n'est pas significative
modele.reduit <- glm(default ~ ., data = bankloans.opt.data[,!(colnames(bankloans.opt.data) 
                                                               == "creddebt")], family = binomial)
#Statistique du rapport de vraisemblance
Sn <- modele.reduit$deviance - modele$deviance
print(Sn)
pvalue.creddebt <- pchisq(q = Sn, df = 1, lower.tail = F)
print(pvalue.creddebt)

#vecteur des p_values
vect.des.pvalues.MV <- c(pvalue.employ, pvalue.address, pvalue.creddebt, pvalue.debtinc)
names(vect.des.pvalues.MV) <- colnames(bankloans.opt.data[,!(colnames(bankloans.opt.data) == "default")])
vect.des.pvalues.MV

sort(vect.des.pvalues.MV) 
sort(vect.des.pvalues.Wald) 
#on obtient le même classement ici! pas toujours le cas ... 
#il est recommandé de retenir le classement selon le test du rapport de vraisemblances.


#########################################################################################
##Scoring##
#########################################################################################




#On partage la base en deux parties : train.set et test.set (avec échantillonnage stratifié)

##Modèle global
library(sampling)
set.seed(12)
mod.default <- as.vector(unique(bankloans.num.data$default))#les modalités de la variable réponse "default"
table(bankloans.num.data$default)[mod.default]
size <- as.vector(table(bankloans.num.data$default)[mod.default]/3) #on veut 1/3 d'observations pour le test 
s <- strata(bankloans.num.data, stratanames="default", size=size, method="srswor")
test.set.index <- s$ID_unit
bankloans.test.set <- bankloans.num.data[test.set.index, ] #ensemble de test
bankloans.train.set <- bankloans.num.data[- test.set.index, ] #ensemble d'apprentissage

#On compare les scores construits par RegLog, lda, qda, forêts aléatoires,
#arbres de décision et svm 
modele.logit <- glm(formula = default ~ ., data = bankloans.train.set, family = binomial)
modele.lda <- lda(formula = default ~., data = bankloans.train.set)
modele.qda <- qda(formula = default ~., data = bankloans.train.set)
modele.RF <- randomForest(formula = default ~., data = bankloans.train.set)
modele.arbre <- rpart(formula = default ~., data = bankloans.train.set)

set.seed(12)
tune.out <- tune(svm, default ~ ., data=bankloans.train.set, kernel="radial", scale=TRUE, 
                 ranges=list(cost = c(2^(-2) ,2^(-1), 1, 2^2, 2^3, 2^4), 
                             gamma=c(2^(-3),2^(-2),2^(-1),1)))
tune.out
modele.svm <- tune.out$best.model

#On calcule ensuite pour chaque modèle le score des individus de l'échantillon de test
Score.logit <- predict(modele.logit, newdata = bankloans.test.set, type = "response")
Score.lda <- predict(modele.lda, newdata = bankloans.test.set, type = "prob")$posterior[, 2]
Score.qda <- predict(modele.qda, newdata = bankloans.test.set, type = "prob")$posterior[, 2]
Score.RF <- predict(modele.RF, newdata = bankloans.test.set, type = "prob")[, 2]
Score.arbre <- predict(modele.arbre, newdata = bankloans.test.set, type = "prob")[, 2]
Score.svm <- attributes(predict(modele.svm, newdata = bankloans.test.set, scale = TRUE, 
                                decision.values = TRUE))$decision.values

#On trace maintenant les 6 courbes ROC 
require(ROCR)
S1.pred <- prediction(Score.logit, bankloans.test.set$default)
S2.pred <- prediction(Score.lda, bankloans.test.set$default)
S3.pred <- prediction(Score.qda, bankloans.test.set$default)
S4.pred <- prediction(Score.RF, bankloans.test.set$default)
S5.pred <- prediction(Score.arbre, bankloans.test.set$default)
S6.pred <- prediction(Score.svm, bankloans.test.set$default)

roc1 <- performance(S1.pred, measure = "tpr", x.measure = "fpr")
roc2 <- performance(S2.pred, measure = "tpr", x.measure = "fpr")
roc3 <- performance(S3.pred, measure = "tpr", x.measure = "fpr")
roc4 <- performance(S4.pred, measure = "tpr", x.measure = "fpr")
roc5 <- performance(S5.pred, measure = "tpr", x.measure = "fpr")
roc6 <- performance(S6.pred, measure = "tpr", x.measure = "fpr")

par(mfrow = c(1,1))

#Tracer les courbes ROC des scores
plot(roc1, col = "black", lwd = 2, main = "Courbes ROC")
plot(roc2, add = TRUE, col = "red", lwd = 2)
plot(roc3, add = TRUE, col = "blue", lwd = 2)
plot(roc4, add = TRUE, col = "green", lwd = 2)
plot(roc5, add = TRUE, col = "yellow", lwd = 2)
plot(roc6, add = TRUE, col = "orange", lwd = 2)
bissect <- function(x) x
curve(bissect(x),  col = "black", lty = 2, lwd = 2, add = TRUE)
legend("bottomright", legend = c("logit", "lda", "qda", "RF", "arbre", "svm"),
       col = c("black", "red", "blue", "green", "yellow", "orange"), lty = 1, lwd = 2)

#Calcul de l'AUC  
AUC1 <- performance(S1.pred, "auc")@y.values[[1]]
AUC2 <- performance(S2.pred, "auc")@y.values[[1]]
AUC3 <- performance(S3.pred, "auc")@y.values[[1]]
AUC4 <- performance(S4.pred, "auc")@y.values[[1]]
AUC5 <- performance(S5.pred, "auc")@y.values[[1]]
AUC6 <- performance(S6.pred, "auc")@y.values[[1]]

print(c("La valeur de l'AUC de chacun des scores : ", "",
        paste("logit = ", as.character(AUC1)),
        paste("lda = ", as.character(AUC2)),
        paste("qda = ", as.character(AUC3)),
        paste("RF = ", as.character(AUC4)),
        paste("arbre = ", as.character(AUC5)),
        paste("svm = ", as.character(AUC6))
)) 

#La courbe Lift de chacun des scores
lift1 <- performance(S1.pred, measure =  "tpr", x.measure =  "rpp")
lift2 <- performance(S2.pred, measure = "tpr", x.measure = "rpp")
lift3 <- performance(S3.pred, measure = "tpr", x.measure = "rpp")
lift4 <- performance(S4.pred, measure = "tpr", x.measure = "rpp")
lift5 <- performance(S5.pred, measure = "tpr", x.measure = "rpp")
lift6 <- performance(S6.pred, measure = "tpr", x.measure = "rpp")

plot(lift1, col = "black", lwd = 2, main = "Courbes Lift")
plot(lift2, add = TRUE, col = "red", lwd = 2)
plot(lift3, add = TRUE, col = "blue", lwd = 2)
plot(lift4, add = TRUE, col = "green", lwd = 2)
plot(lift5, add = TRUE, col = "yellow", lwd = 2)
plot(lift6, add = TRUE, col = "orange", lwd = 2)
bissect <- function(x) x
curve(bissect(x),  col = "black", lty = 2, lwd = 2, add = TRUE)
legend("bottomright", legend = c("logit", "lda", "qda", "RF", "arbre", "svm"),
       col = c("black", "red", "blue", "green", "yellow", "orange"), 
       lty = 1, lwd = 2)



#### le score logit est le meilleur d'après les résulats précédents ####
modele.logit <- glm(formula = default ~ ., data = bankloans.train.set, family = binomial)
Score.logit <- predict(modele.logit, newdata = bankloans.test.set, type = "response")

#voici comment ordonner les individus de test (dans l'ordre décroissant des valeurs du score)
sort(Score.logit, decreasing=TRUE)

#le top dix :
sort(Score.logit, decreasing=TRUE)[1:10]

#Modèle reduit :modèle selectionné avec BIC

set.seed(12)
mod.default <- as.vector(unique(bankloans.opt.data$default))#les modalités de la variable réponse "default"
table(bankloans.opt.data$default)[mod.default]
size <- as.vector(table(bankloans.opt.data$default)[mod.default]/3) #on veut 1/3 d'observations pour le test 
s <- strata(bankloans.opt.data, stratanames="default", size=size, method="srswor")
test.set.index <- s$ID_unit
bankloans.test.set <- bankloans.opt.data[test.set.index, ] #ensemble de test
bankloans.train.set <- bankloans.opt.data[- test.set.index, ] #ensemble d'apprentissage

#On compare les scores construits par RegLog, lda, qda, forêts aléatoires,
#arbres de décision et svm 
modele.logit <- glm(formula = default ~ ., data = bankloans.train.set, family = binomial)
modele.lda <- lda(formula = default ~., data = bankloans.train.set)
modele.qda <- qda(formula = default ~., data = bankloans.train.set)
modele.RF <- randomForest(formula = default ~., data = bankloans.train.set)
modele.arbre <- rpart(formula = default ~., data = bankloans.train.set)

set.seed(12)
tune.out <- tune(svm, default ~ ., data=bankloans.train.set, kernel="radial", scale=TRUE, 
                 ranges=list(cost = c(2^(-2) ,2^(-1), 1, 2^2, 2^3, 2^4), 
                             gamma=c(2^(-3),2^(-2),2^(-1),1)))
tune.out
modele.svm <- tune.out$best.model

#On calcule ensuite pour chaque modèle le score des individus de l'échantillon de test
Score.logit <- predict(modele.logit, newdata = bankloans.test.set, type = "response")
Score.lda <- predict(modele.lda, newdata = bankloans.test.set, type = "prob")$posterior[, 2]
Score.qda <- predict(modele.qda, newdata = bankloans.test.set, type = "prob")$posterior[, 2]
Score.RF <- predict(modele.RF, newdata = bankloans.test.set, type = "prob")[, 2]
Score.arbre <- predict(modele.arbre, newdata = bankloans.test.set, type = "prob")[, 2]
Score.svm <- attributes(predict(modele.svm, newdata = bankloans.test.set, scale = TRUE, 
                                decision.values = TRUE))$decision.values

#On trace maintenant les 6 courbes ROC 
require(ROCR)
S1.pred <- prediction(Score.logit, bankloans.test.set$default)
S2.pred <- prediction(Score.lda, bankloans.test.set$default)
S3.pred <- prediction(Score.qda, bankloans.test.set$default)
S4.pred <- prediction(Score.RF, bankloans.test.set$default)
S5.pred <- prediction(Score.arbre, bankloans.test.set$default)
S6.pred <- prediction(Score.svm, bankloans.test.set$default)

roc1 <- performance(S1.pred, measure = "tpr", x.measure = "fpr")
roc2 <- performance(S2.pred, measure = "tpr", x.measure = "fpr")
roc3 <- performance(S3.pred, measure = "tpr", x.measure = "fpr")
roc4 <- performance(S4.pred, measure = "tpr", x.measure = "fpr")
roc5 <- performance(S5.pred, measure = "tpr", x.measure = "fpr")
roc6 <- performance(S6.pred, measure = "tpr", x.measure = "fpr")

par(mfrow = c(1,1))

#Tracer les courbes ROC des scores
plot(roc1, col = "black", lwd = 2, main = "Courbes ROC")
plot(roc2, add = TRUE, col = "red", lwd = 2)
plot(roc3, add = TRUE, col = "blue", lwd = 2)
plot(roc4, add = TRUE, col = "green", lwd = 2)
plot(roc5, add = TRUE, col = "yellow", lwd = 2)
plot(roc6, add = TRUE, col = "orange", lwd = 2)
bissect <- function(x) x
curve(bissect(x),  col = "black", lty = 2, lwd = 2, add = TRUE)
legend("bottomright", legend = c("logit", "lda", "qda", "RF", "arbre", "svm"),
       col = c("black", "red", "blue", "green", "yellow", "orange"), lty = 1, lwd = 2)

#Calcul de l'AUC  
AUC1 <- performance(S1.pred, "auc")@y.values[[1]]
AUC2 <- performance(S2.pred, "auc")@y.values[[1]]
AUC3 <- performance(S3.pred, "auc")@y.values[[1]]
AUC4 <- performance(S4.pred, "auc")@y.values[[1]]
AUC5 <- performance(S5.pred, "auc")@y.values[[1]]
AUC6 <- performance(S6.pred, "auc")@y.values[[1]]

print(c("La valeur de l'AUC de chacun des scores : ", "",
        paste("logit = ", as.character(AUC1)),
        paste("lda = ", as.character(AUC2)),
        paste("qda = ", as.character(AUC3)),
        paste("RF = ", as.character(AUC4)),
        paste("arbre = ", as.character(AUC5)),
        paste("svm = ", as.character(AUC6))
)) 

#La courbe Lift de chacun des scores
lift1 <- performance(S1.pred, measure =  "tpr", x.measure =  "rpp")
lift2 <- performance(S2.pred, measure = "tpr", x.measure = "rpp")
lift3 <- performance(S3.pred, measure = "tpr", x.measure = "rpp")
lift4 <- performance(S4.pred, measure = "tpr", x.measure = "rpp")
lift5 <- performance(S5.pred, measure = "tpr", x.measure = "rpp")
lift6 <- performance(S6.pred, measure = "tpr", x.measure = "rpp")

plot(lift1, col = "black", lwd = 2, main = "Courbes Lift")
plot(lift2, add = TRUE, col = "red", lwd = 2)
plot(lift3, add = TRUE, col = "blue", lwd = 2)
plot(lift4, add = TRUE, col = "green", lwd = 2)
plot(lift5, add = TRUE, col = "yellow", lwd = 2)
plot(lift6, add = TRUE, col = "orange", lwd = 2)
bissect <- function(x) x
curve(bissect(x),  col = "black", lty = 2, lwd = 2, add = TRUE)
legend("bottomright", legend = c("logit", "lda", "qda", "RF", "arbre", "svm"),
       col = c("black", "red", "blue", "green", "yellow", "orange"), 
       lty = 1, lwd = 2)



#### le score logit est le meilleur d'après les résulats précédents ####
modele.logit <- glm(formula = default ~ ., data = bankloans.train.set, family = binomial)
Score.logit <- predict(modele.logit, newdata = bankloans.test.set, type = "response")

#voici comment ordonner les individus de test (dans l'ordre décroissant des valeurs du score)
sort(Score.logit, decreasing=TRUE)

#le top dix :
sort(Score.logit, decreasing=TRUE)[1:10]

