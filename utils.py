import matplotlib.pyplot as plt 


def plotLearning(scores, filename):   

    plt.figure() 
    plt.plot(scores)
    
    plt.ylabel('Score de Validation')
    plt.xlabel('Épisodes')
    plt.title('Évolution du Score de Validation')
    
    plt.savefig(filename)
    plt.close()