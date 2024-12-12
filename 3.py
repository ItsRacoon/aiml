import numpy as np 
import pandas as pd 
 
data = pd.read_csv("D:/lab/aiml/p3.csv") 
print(data) 

concepts = np.array(data.iloc[:, 0:-1]) 
print(concepts) 

target = np.array(data.iloc[:, -1]) 
print(target) 
 

def learn(concepts, target):  
    specific_h = concepts[0].copy() 
    print("\nInitialization of specific_h and general_h") 
    print("\nSpecific hypothesis: ", specific_h) 
    
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))] 
    print("\nGeneric hypothesis: ", general_h)   
    
    for i, h in enumerate(concepts): 
        print("\nInstance", i + 1, "is", h) 
        
        if target[i] == "yes": 
            print("Instance is Positive") 
            for x in range(len(specific_h)):  
                if h[x] != specific_h[x]:                     
                    specific_h[x] = '?'                      
                    general_h[x][x] = '?' 
               
        if target[i] == "no":             
            print("Instance is Negative") 
            for x in range(len(specific_h)):  
                if h[x] != specific_h[x]:                     
                    general_h[x][x] = specific_h[x]                 
                else:                     
                    general_h[x][x] = '?'         
        
        print("Specific hypothesis after", i + 1, "Instance is", specific_h)          
        print("Generic hypothesis after", i + 1, "Instance is", general_h) 
        print("\n") 
    
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]     
  
    for i in indices:    
        general_h.remove(['?', '?', '?', '?', '?', '?'])  
    
    return specific_h, general_h 
 
s_final, g_final = learn(concepts, target) 
 
print("Final Specific_h:", s_final, sep="\n") 
print("Final General_h:", g_final, sep="\n")
