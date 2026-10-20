# %%
import pandas as pd
import numpy as np

df = pd.read_csv('messy_house_prices.csv')
print(f" starting row_count: { len(df)} rows")


df['SquareFootage'] = df['SquareFootage'].astype(str).str.replace(',', '')
df["BHK"] = df['BHK'].astype(str).str.extract(r'(\d+)', expand = False)


df['SquareFootage'] = pd.to_numeric(df['SquareFootage'], errors='coerce')
df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')
df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')
df['Price_Lakhs'] = pd.to_numeric(df['Price_Lakhs'], errors='coerce')





df_clean = df.dropna()
print(f" starting row_count after data cleaning: { len(df_clean)} rows")
#print(df_clean.info)

df_clean.to_csv("cleaned_indian_houses.csv", index=False)
print("Updated file was save succecssfully")

X = df_clean[['SquareFootage', 'BHK', 'Bathrooms']].values
Y = df_clean['Price_Lakhs'].values

X_min  = np.min(X, axis=0)
X_max = np.max(X, axis=0)

#scaled data
X_scaled = (X- X_min) / (X_max - X_min)

#print(X_max, X_min, X_scaled[0]) 
print(X_scaled.shape)



#model
# .shaped gives m=rows, n=cols,, for training eg
m, n = X_scaled.shape

#initial weights of our varaibles, and bias constanr
weights = np.zeros(n)
bias = 0.0
#print(f"training samples of m:{m}")
#print(f"no of input varaibles: {n}")

iterations = 5000
learning_rate  = 0.01
cost_history=[]
iteration_history=[]

for i in range(iterations):
    
    #Y=X.W+B    
    y_model = np.dot(X_scaled, weights) + bias

    #print(f"y_model shape: {y_model.shape}")
    error = y_model - Y
    
    #partial derivatives(gradients)
    dJ_dw = (1/m)* np.dot(X_scaled.T,error)
    dJ_db = (1/m)*np.sum(error)
    
    #update step, simulatneous adjustment
    weights = weights - learning_rate*dJ_dw
    bias = bias - learning_rate*dJ_db
    
    current_cost = (1/(2*m))*(np.sum(error**2))
    cost_history.append(current_cost)
    iteration_history.append(i)
    #check error reducing here
    if i % 1000 == 0:
        print(f"iteratin {i:4d} | current cost:{current_cost:.4f} ")


#last at 10000 iter error
final_cost_error = cost_history[-1]

rmse = np.sqrt((2 * final_cost_error))


#optimizatiin complete!
print(f"final weights (w1, w2, w3): {weights}")
print(f"bias is : {bias}")
print(f"Final Root Mean Squared Error (RMSE):{rmse} lakhs")




#LIVE USER IMPUT EVELUATION
def predict_house_price(raw_sqft, raw_bhk, raw_bath):
    raw_input = np.array([raw_sqft, raw_bhk, raw_bath])
    scaled_input = (raw_input - X_min)/(X_max-X_min)
    
    predicted_price = np.dot(scaled_input, weights) + bias
    
    return predicted_price



#Test with real nums
price1 = predict_house_price(1800,3,3)
print(f"price-1 for house with 1800sqft, 3BHK, 3bath : {price1:.2f} Lakhs") 


import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))
plt.plot(iteration_history, cost_history, color='crimson', linewidth=2.5, label="Cost Function J(w,b)")
plt.title('Gradient Descent Learning Curve (Error reduction Progress)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Number of iterations", fontsize=11)
plt.ylabel("Cost value(Mean Sqaure Error)", fontsize=11)
plt.grid(True, linestyle='--', alpha =0.5)
plt.legend(fontsize=11)
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
