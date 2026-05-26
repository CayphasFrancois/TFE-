import matplotlib.pyplot as plt 


def plotLearning(scores):   

    plt.figure() 
    plt.plot(scores)
    
    plt.ylabel('Score de Validation')
    plt.xlabel('Épisodes')
    plt.title('Évolution du Score de Validation')
    

    plt.close()
