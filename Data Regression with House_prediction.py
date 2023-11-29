from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression ,LinearRegression
from sklearn.metrics import r2_score, confusion_matrix,mean_squared_error,accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib as plt

# 1. Lire et normaliser les données
# Supposons que data est votre DataFrame contenant les données immobilières
data = pd.read_csv("Datasets//House_prediction.txt")

x=data.drop('prix', axis=1)
y=data["prix"]

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalisation 
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

# 2. Régression linéaire multiple
model = LinearRegression()
model.fit(X_train_normalized, y_train)
y_val_pred = model.predict(X_val_normalized)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)


# 3. Régression polynomiale

best_degree = None
best_mse = float('inf')
best_r2 = -float('inf') 

for degree in range(1, 8):
    
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train_normalized)
    X_val_poly = poly.transform(X_val_normalized)
    
    #training, predict
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_val_pred = model.predict(X_val_poly)

    #MSE 
    mse = mean_squared_error(y_val, y_val_pred)
    #R2
    r2 = r2_score(y_val, y_val_pred)

    print(f'Degré {degree}:',f' MSE = {mse},',f' R2 = {r2}')

    # BEST MODEL
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_degree = degree
print('best degree =',best_degree)

# 4. Ordre optimal du modèle

def poly(degree,x,y):
        
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train) #will be used f training
    X_val_poly = poly.transform(X_val) #will be used f prediction
    X_test_poly = poly.transform(X_test)
    
    # Entrainement
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # prediction
    y_val_pred = model.predict(X_val_poly)
    y_test_pred = model.predict(X_test_poly)
    
    
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print('mse =',mse_test,'\n','r2 =',r2_test)
    
poly(best_degree,x,y)



# Insérez ici le code pour prédire les données de test avec le meilleur modèle polynomial
# et évaluer l'erreur quadratique moyenne et le coefficient de détermination.

'''-----------------------------------5.2----------------------------------------------------------'''


data3 = pd.read_csv("C:/Users/helfo/Downloads/Telco-Customer-Churn.csv", index_col=None)


X_train, X_test, c_train, c_test = train_test_split(data3['tenure'].values.reshape(-1, 1), data3['Churn'], test_size=0.3, random_state=42)

def sagReg(X_train, X_test, c_train, c_test):
    model_sag = LogisticRegression(solver='sag')
    model_sag.fit(X_train, c_train)
    
    # Prédiction des probabilités 
    probs_app_sag = model_sag.predict_proba(X_train)
    probs_test_sag = model_sag.predict_proba(X_test)
    
    # Prédiction des classes
    cp_app = model_sag.predict(X_train)
    cp_test = model_sag.predict(X_test)
    
    # Calcul des scores 
    score_app_sag = model_sag.score(X_train, c_train)
    score_test_sag = model_sag.score(X_test, c_test)
    
    # matrice de confusion entre c_app et la classe prédite noté cp_app
    confusion_app_sag = confusion_matrix(c_train, cp_app)
    
    # Calcul de la matrice de confusion entre c_test et la classe prédite noté cp_test
    confusion_test_sag = confusion_matrix(c_test, cp_test)
    
    # Matrice de confusion pour SAG
    print("Matrice de confusion de test(SAG):")
    print(confusion_test_sag,"\n")
    
    print("Matrice de confusion d'apprentissage SAG:")
    print(confusion_app_sag,"\n")
    
    # Calcul de la précision (accuracy) pour SAG
    accuracy_sag = (confusion_test_sag[0, 0] + confusion_test_sag[1, 1]) / np.sum(confusion_test_sag)
    print(f"Précision (SAG): {accuracy_sag:.2f}\n")
    
    #True Positive Rate SAG
    tpr_sag = confusion_test_sag[1, 1] / (confusion_test_sag[1, 0] + confusion_test_sag[1, 1])
    print(f"Taux de Vrais Positifs (SAG): {tpr_sag:.2f}\n")
    
    #False Positive Rate SAG
    fpr_sag = confusion_test_sag[0, 1] / (confusion_test_sag[0, 0] + confusion_test_sag[0, 1])
    print(f"Taux de Faux Positifs (SAG): {fpr_sag:.2f}\n")

sagReg(X_train, X_test, c_train, c_test)

'''-_-_-_-_- NEWTON-CG-_-_-_-_-_-_-_-_-'''

def newton(X_train, X_test, c_train, c_test):
    # Régression logistique avec le solveur Newton-CG
    model_newton_cg = LogisticRegression(solver='newton-cg')
    model_newton_cg.fit(X_train, c_train)

    # Prédiction des probabilités 
    probs_app_newton_cg = model_newton_cg.predict_proba(X_train)
    probs_test_newton_cg = model_newton_cg.predict_proba(X_test)

    # Prédiction des classes
    cp_app = model_newton_cg.predict(X_train)
    cp_test = model_newton_cg.predict(X_test)

    # Calcul des scores 
    score_app_newton_cg = model_newton_cg.score(X_train, c_train)
    score_test_newton_cg = model_newton_cg.score(X_test, c_test)

    # Calcul de la matrice de confusion entre c_app et la classe prédite notée cp_app
    confusion_app_newton_cg = confusion_matrix(c_train, cp_app)
    # Calcul de la matrice de confusion entre c_test et la classe prédite noté cp_test
    confusion_test_newton_cg = confusion_matrix(c_test, cp_test)
    # Matrice de confusion pour Newton-CG
    print("Matrice de confusion (Newton-CG):")
    print(confusion_app_newton_cg,'\n')
    print("Matrice de confusion (Newton-CG):")
    print(confusion_test_newton_cg,'\n')
    
    # Calcul de la précision (accuracy) pour Newton-CG
    accuracy_newton_cg = (confusion_test_newton_cg[0, 0] + confusion_test_newton_cg[1, 1]) / np.sum(confusion_test_newton_cg)
    print(f"Précision (Newton-CG): {accuracy_newton_cg:.2f}\n")
    
    #True Positive Rate newton
    tpr_newton_cg = confusion_test_newton_cg[1, 1] / (confusion_test_newton_cg[1, 0] + confusion_test_newton_cg[1, 1])
    print(f"Taux de Vrais Positifs (Newton-CG): {tpr_newton_cg:.2f}\n")
    
    #false Positive Rate newton
    fpr_newton_cg = confusion_test_newton_cg[0, 1] / (confusion_test_newton_cg[0, 0] + confusion_test_newton_cg[0, 1])
    print(f"Taux de Faux Positifs (Newton-CG): {fpr_newton_cg:.2f}\n")

newton(X_train, X_test, c_train, c_test)


X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(data3['tenure'].values.reshape(-1, 1), data3['Churn'], test_size=0.3, random_state=42)

# Application de la régression logistique
# Changez 'solver' selon votre choix : 'lbfgs', 'newton-cg', 'sag'
def lbfg(X_train_logistic, y_train_logistic,X_test_logistic,y_test_logistic):
    logistic_model = LogisticRegression(solver='lbfgs')
    logistic_model.fit(X_train_logistic, y_train_logistic)  # Choisissez une seule caractéristique (xi)
    
    y_pred_logistic = logistic_model.predict(X_test_logistic)
    confusion = confusion_matrix(y_test_logistic, y_pred_logistic)
    print(confusion)
    accuracy = accuracy_score(y_test_logistic, y_pred_logistic)
    print(accuracy)

    
lbfg(X_train_logistic, y_train_logistic,X_test_logistic,y_test_logistic)


