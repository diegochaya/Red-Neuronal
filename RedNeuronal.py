import numpy as np
import matplotlib.pyplot as plt

class RedNeuronal():
    def __init__(self,layers,fun):
      self.layers=layers
      self.n=len(self.layers)
      self.P=[]
      self.b=[]
      self.func=[0]*(self.n)
      self.derfunc=[0]*(self.n)
      for i in range(self.n):
        if fun[i]=="sigmoid":
          self.func[i]=lambda x: 1 / (1 + np.exp(-x))
          self.derfunc[i]=lambda x: np.exp(-x) / (1 + np.exp(-x))**2
        elif fun[i]=="sin":
          self.func[i]=lambda x: np.sin(x) ** 2
          self.derfunc[i]=lambda x: np.sin(2*x)
        elif fun[i]=="id":
          self.func[i]=lambda x: x
          self.derfunc[i]=lambda x: np.ones_like(x)
        elif fun[i]=="relu":
          self.func[i]=lambda x: np.maximum(0, x)
          self.derfunc[i]=lambda x: np.where(x > 0, 1, 0)
        else:
          raise Exception("Función "+ str(i)+" no reconocida")
      for i in range(self.n - 1):
          self.P.append(np.random.rand(layers[i], layers[i + 1]) - 0.5)
          self.b.append(np.random.rand(1, layers[i + 1]) - 0.5)

    def entrenar_pesos(self, M, y, max_iters, nu, epsilon,k=20,nu_var=1):
        #M np.Matriz con los datos de entrenamiento
        #y np.Matriz con los resultados de entranamiento
        ndat = len(M)
        Lant=np.inf
        for et in range(max_iters):
            derP = []
            derb = []
            L=0
            #for idat in range(ndat):
            Z = [np.array(M)]
            for i in range(self.n - 1):
                Z.append(self.func[i](Z[-1]).dot(self.P[i]) + self.b[i])
            L += np.sum((self.func[-1](Z[-1]) - y) ** 2)
            dP = []
            db = []
            derA = 2 * (self.func[-1](Z[-1]) - y)
            for l in range(self.n - 1, 0, -1):
                dP = [self.func[l-1](Z[l-1]).T.dot((self.derfunc[l](Z[l]) * derA))] + dP
                db = [np.sum(self.derfunc[l](Z[l]) * derA,axis=0,keepdims=True)] + db
                derA = (derA * self.derfunc[l](Z[l])).dot(self.P[l-1].T)
            for i in range(self.n - 1):
                self.P[i] -= nu * dP[i]
                self.b[i] -= nu * db[i]
            if Lant<=L:
              if nu_var!=1:
                nu*=nu_var
                print("nu cambiado: "+str(nu))
              else:
                print("No converge")
                break
            Lant=L
            if et%k==0:
              print(et,L)

            if L < epsilon:
                break

    def predecir(self,x, Clasificar=0):
        #Clasificar = 0, no se clasifica
        #Clasificar = 1, se clasifica binariamente
        #Clasificar = 2, se selecciona el mejor resultado
        result = self.func[0](x)
        for i in range(self.n-1):
            result = self.func[i+1](result.dot(self.P[i]) + self.b[i])
        if Clasificar:
            if Clasificar==1:
                result = np.where(result > 0.5, 1, 0)
            elif  Clasificar==2:
                result = np.argmax(result)
        return result

    def EscribirPesos(self,filename="EntrenamientoImagenes.npz"):
      matrices={f"P{i}": m for i, m in enumerate(self.P)}
      matrices.update({f"b{i}": m for i, m in enumerate(self.b)})
      np.savez(filename,**matrices)

    def LeerPesos(self,filename="EntrenamientoImagenes.npz"):
      datos=np.load(filename)
      self.P=[]
      self.b=[]
      for i in range(self.n-1):
        if datos[f"P{i}"].shape==(self.layers[i],self.layers[i+1])and datos[f"b{i}"].shape==(1,self.layers[i+1]):
          self.P.append(datos[f"P{i}"])
          self.b.append(datos[f"b{i}"])
        else:
          raise Exception("Dimensiones de los pesos no coinciden")

    def GraficarPesos(self):
      for i in range(self.n-1):
        plt.subplot(1,self.n,i+1)
        plt.imshow(self.P[i], cmap="viridis", interpolation="nearest")
        plt.colorbar()  # Agregar barra de colores
        plt.title("P"+str(i))
      plt.show()
