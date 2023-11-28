from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression ,LinearRegression
from sklearn.metrics import r2_score, confusion_matrix,mean_squared_error,accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''------------------------------------------------------------------------------------------'''
data = pd.read_csv("C:/Users/helfo/Downloads/DataRegression.csv")
x = data[['x1', 'x2', 'x3']]
y =data['y']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

'''-------------------------------Régression continue-------------------------------------------------'''
#1
correlations = data.corr()
print("Corrélations :\n",correlations)
#On remarque que x1 a la plus grande correlation avec y

#2
#X_train, X_test, y_train, y_test = train_test_split(data['x1'],data['y'], test_size=0.2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data['x1'], data['y'], test_size=0.2, random_state=42)

#3
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#4
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Erreur Quadratique Moyenne :", mse,'\n')
print("Coefficient de Détermination :", r2,'\n')


# Calcul de MSE (version locale)

n = len(y_test)
mse = 0
for i in range(n):
    mse += ((y_test.values[i] - y_pred[i]) ** 2) /n
print('MSE (without functions) :',mse,'\n')

# Calcul du coefficient de détermination (version locale)

y_mean = sum(y_test) / n
sst = 0
for true in y_test:
    sst += (true - y_mean) ** 2

sse = 0
for i in range(n):
    sse += (y_test.iloc[i] - y_pred[i]) ** 2

r2 = 1 - (sse / sst)
print(" R2 (without functions): ", r2)


#5
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Régression linéaire')
plt.legend()
plt.show()



'''-_-_-_-_-_6) NOW USING X1 X2 X3 REGRESSION LINEAIRE MULTIPLE -_-_-_-_-_-'''

x = data[['x1', 'x2', 'x3']]
y =data['y']


X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(x, y, test_size=0.2, random_state=42)
model_mult = LinearRegression()
model_mult.fit(X_train_mult, y_train_mult)
#print(model_mult.score(x, y)) determination coef
y_pred_test_mult = model_mult.predict(X_test_mult)


mse = mean_squared_error(y_test_mult, y_pred_test_mult)
r2 = r2_score(y_test_mult, y_pred_test_mult)
print("Erreur Quadratique Moyenne :", mse,'\n')
print("Coefficient de Détermination :", r2,'\n')


# Calcul de MSE (version locale)
n = len(y_test_mult)
mse = 0
for i in range(n):
    mse += ((y_test_mult.values[i] - y_pred_test_mult[i]) ** 2) /n
print('MSE (without functions) :',mse,'\n')

# Calcul du coefficient de détermination (version locale)
y_mean = sum(y_test) / n
sst = 0
for true in y_test:
    sst += (true - y_mean) ** 2
sse = 0
for i in range(n):
    sse += (y_test.iloc[i] - y_pred[i]) ** 2
r2 = 1 - (sse / sst)
print("Coefficient de Détermination (without functions): ", r2)

# Tracer les données réelles
plt.figure(2)
plt.scatter(X_test_mult['x1'], y_test_mult, color='b', label='x1 - Données réelles')
plt.scatter(X_test_mult['x2'], y_test_mult, color='g', label='x2 - Données réelles')
plt.scatter(X_test_mult['x3'], y_test_mult, color='r', label='x3 - Données réelles')

# Tracer les prédictions
plt.plot(X_test_mult['x1'], y_pred_test_mult, color='c', linewidth=2, label='x1 - Régression linéaire')
plt.plot(X_test_mult['x2'], y_pred_test_mult, color='m', linewidth=2, label='x2 - Régression linéaire')
plt.plot(X_test_mult['x3'], y_pred_test_mult, color='y', linewidth=2, label='x3 - Régression linéaire')
plt.xlabel('Valeurs réelles de y')
plt.ylabel('Prédictions de y')
plt.legend()
plt.title('Régression linéaire - Valeurs réelles vs Prédictions')
plt.show()

#function to import apres
def regMult(x,y):
    
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)
    model_reg_mul=LinearRegression()
    model_reg_mul.fit(X_train,y_train)
    y_pred_mult_reg=model_reg_mul.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_mult_reg)
    r2 = r2_score(y_test, y_pred_mult_reg)
    print("Erreur Quadratique Moyenne :", mse,'\n')
    print("Coefficient de Détermination :", r2,'\n')
regMult(x,y)


'''-_-_-_-_-_7-8-9) REGRESSION POLYNOMIALE-_-_-_-_-_-'''

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

best_degree = None
best_mse = float('inf')
best_r2 = -float('inf') 

for degree in range(2, 11):
    
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
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


#fonction pour faire la regression poly 
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



# # Plot des données réelles et des prédictions
# plt.scatter(y_test, y_test_pred, color='blue', label='Données réelles vs Prédictions')
# plt.plot(X_test, y_pred, color='red', linewidth=2, label='Régression linéaire')
# plt.xlabel('Valeurs réelles')
# plt.ylabel('Prédictions')
# plt.title('Prédictions vs Données Réelles (Régression polynomiale de degré 2)')
# plt.legend()
# plt.show()


'''-------------------------------Régression discrete------------------------------------------------'''
'''-_-_-_-_-2)Régression logistique with SAG-_-_-_-_-_-_-_-_-'''

X_train, X_test, c_train, c_test = train_test_split(data['x1'].values.reshape(-1, 1), data['c'], test_size=0.3, random_state=42)

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

'''-_-_-_-_-2)Régression logistique with NEWTON-CG-_-_-_-_-_-_-_-_-'''

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
    print("Matrice de confusion apprentissage (Newton-CG):")
    print(confusion_app_newton_cg,'\n')
    print("Matrice de confusion test (Newton-CG):")
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

'''-------------------------------5---------------------------------------------------------------'''