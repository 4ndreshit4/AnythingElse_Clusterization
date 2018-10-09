# AnythingElse_Clusterization
This project looks for manage all inputs that Tabot hasn´t recognize. 


Proceso de clusterización de las entradas clasificadas como "Anything Else" (Not handled) del agente virtual Tabot



#-----------------------------------------------------------------------------------------------------------------------------
TABOT

El agente virtual Tabot (Chatbot con front en Messenger Facebook) tiene en sus capacidades responder a asesorias, transacciones y notificaciones que brinden de manera continua, atención a las necesidades puntuales de los clientes sobre productos ya adquiridos o información general sobre productos del banco. (Consultas de saldo,Tarjetas, TRM, Sucursales, entre otros.)

Con el fin de que Tabot sea un canal más atractivo para el usuario y realmente se convierta en un aliado para él, es requerido que éste, esté entrenado de manera tal que pueda entender y discernir entre diferentes temas de interés. En algunos casos, estos tópicos pueden estar fuera del eje temático en torno al negocio del banco o pueden ser nuevas tendencias y casos de utilidad.

Multiples usuarios día a día interactúan con Tabot, la pluralidad de mensajes, preguntas y solicitudes varían enormemente, usualmente en en relación con quien es ese usuario, sus caracteristicas demográficas y la relación comercial que tenga con el banco. El interés del público por "testear" al chatbot y la simple curiosidad de quienes le consultan, hace que una gran cantidad de solicitudes se queden sin responder. Ya que Tabot tiene actualmente un campo limitado de comprensión.

Por ejemplo, la información solicitada cuando se pregunta por -el equipo de futbol favorito o un chiste casual - no pertence a los dominios entrenados de Tabot. Estos atributos un poco mas << humanos >> aún no se le han enseñado, aunque ya se han estado identificando. La humanización en el flujo del lenguaje de los chatbot permiten una mejor relacion con los usuarios y propone nuevos entornos de confianza y aceptación. De manera similar, existen otros intereses que los usuarios han ingresado, y requieren revisión ya que aún no se han caracterizado en el dominio de Tabot y esto es un gran punto de mejora.

Con el fin de potencializar las capacidades en el entendimiento de los dominios de Tabot y aumentarlos, se plantea una revisión profunda, de esas solicitudes y preguntas que los clientes han ingresado y que aún Tabot desconoce respuesta o razón alguna. A estos ingresos se les conoce como frases Anything Else, porque de este modo es como las clasifica Watson Assistant, la capacidad cognitiva sobre la que está soportada el bot (Click para más info: https://goo.gl/fMLExk). Por tal razón se propone indagar en esta información, y buscar tendencias que permitan conocer los diferentes campos de interes de los usuarios e incluirlos en Tabot.


#-----------------------------------------------------------------------------------------------------------------------------
Ruta de indagación:

    Clusterizar gran parte de las frases Anything Else ingresadas por los usuarios a Tabot de acuerdo a sus similitudes y relaciones semánticas.
    Hipotesis inicial: Organizar estas frases en cuatro grupos:
        Frases AE que hacen mención a alguna intención ya entrenada en Tabot y que éste no pudo reconocer. (Bugs)
        Frases AE que hacen mención a temas asociados con el negocio del Banco, las finanzas y la economia y no están incluidas en el corpus de Tabot.
        Frases AE que hagan mención a temas más << humanos >>, comunes entre el trato de las personas y la comunicación cotidiana.
        Ingresos cuyo contenido no sean apropiados para considerarse de relaciones humanas ni tampoco hagan referencia a temas relacionados con el Banco y sus áreas de acción, y que por lo tanto, no interesan.

#-----------------------------------------------------------------------------------------------------------------------------
Frases AE: Los datos

Las interacciones entre los usarios y Tabot quedan registradas en Messenger Facebook, que a través de Graph API permite la extracción de datos, además, de distitos parámetros propios de la interacción. (Click para mas info: https://goo.gl/rh7gib). De Graph API se descargó un set de datos correspondiente a las interaciones (mensajes,usuarios y fechas, entre otros) comprendidas entre ENERO 1 y JUNIO 5 de 2018. De allí se logró extraer los mensajes no reconocidos por Tabot, llamados frases Anything Else.(Ver fichero). En total se obtuvieron 41.232 frases. Acá tenemos nuestro set de datos.

#-----------------------------------------------------------------------------------------------------------------------------
Metodologías

En el contexto de NLU (Natural Lenguaje Understanding) y el NLP (Natural Lenguaje Processing) existen distintos metodos de computación que permiten una caracterizacion del lenguaje natural para limpiarlo, analizarlo, encontrar patrones y tendencias de grandes cantidades de textos y por ultimo realizar modelos que sirvan de sistemas de caracterizacion automática. Métodos usados:

- Tokenization:Esto significa convertir el texto en bruto en una lista de palabras.El texto limpio casi siempre significa una lista de palabras o tokens con los que se puede trabajar en modelos de aprendizaje automático.
- Word embedding: Algoritmos para la transformación de palabras del Lenguaje Natural a números.En este proyecto se usa el algoritmo de vectorización Word2Vec, un moderno desarrollo hecho por Google para facilitar distintos computos  de Lenguaje Natural. (Click para más info: https://code.google.com/p/word2vec/ ) 
- K-means: Algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en k grupos basándose en sus características. 

En este sentido, se abordan estos procedimientos de modo tal que se pueda dar una solución y agrupar la información obtenida de las frases AE.

Referencias bibliográficas:

    Deep Learning for Natural Language Processing (Jason Brownlee)

#-----------------------------------------------------------------------------------------------------------------------------
Python: Librerías y recursos

Sklearn: Scikit-learn es la principal librería que existe para trabajar con Machine Learning en Python, incluye la implementación de un gran número de algoritmos de aprendizaje automático supervizado y no supervizado, entre los cuales se encuentra el modelo K-means. (Click para mas info: http://scikit-learn.org)

Gensim: Biblioteca de código abierto de Python para el procesamiento de lenguaje natural, con un enfoque en el modelado de temas. (NLU) (Click para más info: https://radimrehurek.com/gensim/)

Numpy: Librería que provee herramientas matemáticas para trabajar con arrays multidimensionales. (Click para más info: http://www.numpy.org/)

Regular Expressions: Las expresiones regulares se utilizan para identificar si existe un patrón en una secuencia de caracteres determinada (cadena) o no. Ayudan en la manipulación de datos textuales. (Click para más info: https://docs.python.org/3/library/re.html)

#-----------------------------------------------------------------------------------------------------------------------------

Código:


# Include libraries

    from sklearn.cluster import KMeans 

    from gensim.models import Word2Vec

    import numpy as np

    import re

#-----------------------------------------------------------------------------------------------------------------------------
Importación de los < datos >: Se carga el archivo que contiene las frases Anything Else y se empaquetan en una única lista

with open('Anything_Else.txt', 'rt',encoding='utf-8') as File_Data:

    vAnythingElse = [] # List of Anything Else Phrases  

    for row in File_Data:

         vAnythingElse.append(row)  

#-----------------------------------------------------------------------------------------------------------------------------
Stop Words: Escencialmente son palabras de conexión entre las palabras útiles y preposiciones no representan un contenido relevante de la información. (Ejemplos: de,la,que,el,en,y,a,los,del,se,las,por,un,para, entre otras.) (Ver fichero: https://github.com/Alir3z4/stop-words) El objetivo es quitar estas StopWords de los textos que componen las frases AE y analizar las palabras útiles.

Importación de las < Stop Words >

    with open('spanish.txt', 'rt',encoding='utf-8') as Stop_Words:

         vStopWords = []# List of stop or ignore words 

         for row in Stop_Words:

             row = row.replace('\n', '')   

             vStopWords.append(row)

#-----------------------------------------------------------------------------------------------------------------------------
Definición de la función Tokenizer_Sentences: Transforma una frase en una lista de las palabras de las que se compone la frase

    def fn_tokenize_sentences(sentence): 

        """

        Extract the words from a text, 

        then separate each word like a single object and finally makes words lower case

        """

        vWords = re.sub("[^\S]"," ",sentence).split() #nltk.word_tokenize(sentence)

        vWordsToken = [w.lower() for w in vWords]

        return(vWordsToken) 

#-----------------------------------------------------------------------------------------------------------------------------
Definición de la función Stop_Word_Clean: Elimina las StopWords de una lista de palabras

    def fn_stop_words_clean(vListOfWords):

        """

        Wait for a list of words to clean all the "stopwords" in it

        """

        vCleanedList= []

        for word in vListOfWords:

            if word not in vStop_words:

               vCleanedList.append(word)

        return(vCleanedList)

#-----------------------------------------------------------------------------------------------------------------------------
Definición de la función Counting_Words:

    def fn_counting_words(_vIdTokenphrase):

        """

        How many words exists in the list of AE phrases and how many are unique

        """

        vTotalWords = []

        for row in _vIdTokenphrase:

            for word in row[1]:

                vTotalWords.append(word)

        vLenghtTotalWords = len(vTotalWords)

        vUniqueWords = len(set(vTotalWords))

        return(vLenghtTotalWords,vUniqueWords) 

#-----------------------------------------------------------------------------------------------------------------------------
Definición de la función Fix_the_data

    def fn_fix_the_data():

        """"

        It takes the AE phrases as income. Then it apply the fn_tokenize_sentences  and the fn_stop_words_clean functions 

        to obtain as result a list of each tokenized AE phrase and its respective ID.

        """"

        vIdTokenPhrase = []        

        for i in range(len(vAnythingElse)):        

            vWordTokenized= fn_tokenize_sentences(vAnythingElse[i])

            vWordWhitoutStopWords = fn_stop_words_clean(vWordTokenized)    

            vIdTokenPhrase.append([i,vWordWhitoutStopWords])

        return(vIdTokenPhrase)


#-----------------------------------------------------------------------------------------------------------------------------
Definición de la funcion Training_the_model: En esta función se toman todas las palabras del set de datos de las frases AE y en relación con la frecuencia de aparición de cada palabra, y la ubicación relativa en conexión con otras palabras, se realiza la generación de un espacio vectorial donde cada palabra toma un valor asignado de acuerdo a las caracteristicas descritas anteriormente. Para éste procedimiento de descartó la posiblidad de utilizaer un modelo pre-entrenado (Set de palabras del castellano genéricas o universales) debido a la importancia de incluir todas esas palabras que fueron ingresadas por la comunidad de Tabot: Distintas personas de Colombia con una jerga específica de la región. De modo que el espacio vectorial se hizo a partir solo de las palabras halladas en el set de AE

    def fn_training_the_model():

        """"

        The word2vec tool takes a text corpus as input and produces the word vectors as output. 

        It first constructs a vocabulary from the training text data and then learns vector representation of words. 

        Resource: https://code.google.com/archive/p/word2vec/

        """

        vIdTokenPhrase_ = fn_fix_the_data()

        vDataTrain = [] # Define training data/// Anything Else data set

        for id_prhase in vIdTokenPhrase_:

                vDataTrain.append(id_prhase[1])

        vModel = Word2Vec(vDataTrain, min_count=1,size=150,window=5,workers=4) # train model method

        vModel.save('model_word2vect_ae.bin') # save the model: The model can be stored/loaded

Word2vect - Parameters:

Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=())

sentences: iterable of iterables. The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network.

min_count : int. Ignores all words with total frequency lower than this.

size : int. Dimensionality of the feature vectors.

window : int. The maximum distance between the current and predicted word within a sentence.

workers : int. Use these many worker threads to train the model (=faster training with multicore machines).

sg : int {1, 0} Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.

#-----------------------------------------------------------------------------------------------------------------------------
Definición de la función Load_the_model: Una vez entrenado un modelo con los parámetros establecidos, se puede guardar para cargarse y ser utilizado en cualquier parte del proceso.

    def fn_load_the_model():

        # load model

        vLoadModel = Word2Vec.load('model_word2vect_ae.bin') # The model can be stored/loaded

        print("\nFeature's Model: ",vLoadModel)

        print('\n')

        return(vLoadModel)

Start of process

#-----------------------------------------------------------------------------------------------------------------------------
# Initial data parameters visualization


print('\nAnything Else Phrases:',len(vAnythingElse)) # Amount of AE phrases 

vIdTokenPhrase_ = fn_fix_the_data()

vLenghtTotalWords_ , vUniqueWords_ = fn_counting_words(vIdTokenPhrase_)

print('Total words:',vLenghtTotalWords_) 

vPercentageUniqueWords = ((vUniqueWords_ * 100) / vLenghtTotalWords_)

print('Unique words:',vUniqueWords_, '(' , "%.2f" % vPercentageUniqueWords ,'%)')

out []:

 - Anything Else Phrases: 41232
 - Total words: 81779
 - Unique words: 14202 ( 17.37 %)

vModelBOW = fn_load_the_model()

out []:

  - Feature's Model:  Word2Vec(vocab=14202, size=150, alpha=0.025)

vocab : Int. Amount of vectorized words

size: Int. Dimensionality of the feature vectors.

alpha : Float. The initial learning rate.

Método para hacer una lista del ID de cada frase más el equivalente vector de cada una de las palabras que la compone

#------------------------------------------------------------------------------

# Convert each word of AE phrases in a vector by word:

vArrayWord2vec = [] # List of Id + vectors of words <list>

for element in vIdTokenPhrase_:

    vAE_Sentence = element[1]

    vVectors= []

    vVectors.clear()

    for word in vAE_Sentence:

        vToVector = vModelBOW[word]  # get a vector for each word

        vVectors.append(vToVector)

    vArrayWord2vec.append([element[0],vVectors])



#-----------------------------------------------------------------------------------------------------------------------------
Procedimiento para sumar los vectores que componen la frase y de este modo obtener un vector resultante por cada frase



# Convert all vectors in just one through sum of vectors

vArraySumOfVectors = [] #  List of Id + sum of vectors

for ae_vector in vArrayWord2vec:

    vSum_Vec = np.zeros((150,), dtype='float32')

    for i in range(len(ae_vector[1])):

        vSum_Vec  = vSum_Vec + ae_vector[1][i]

    vArraySumOfVectors.append([ae_vector[0],vSum_Vec]) # Id , vect_sum ===> AE phrases <=====

    

#-----------------------------------------------------------------------------------------------------------------------------
Modelo de clusterización: K- means

Se definen 4 cluster (K = 4) basandose en la hipótesis inicial de agrupación



# K-means model: Clusterization

vXData = [] # Data set AE

for row in vArraySumOfVectors: 

    vXData.append(row[1])

vClusterNum = 4

k_means = KMeans(init = "k-means++", n_clusters = vClusterNum, n_init = 12)

k_means.fit(vXData)

vLabels = k_means.labels_



print('\nNumber of phrases labeled:',len(vLabels))

print('Labels:', set(vLabels))



out[]:

      - Number of phrases labeled: 41232

      - Labels: {0, 1, 2, 3}

     
El modelo devuelve una lista ordenada de etiquetas que corresponde al nombre del cluster en el que se agrupó cada frase. De allí se puede extraer la cantidad de frases que fueron etiquetadas, y el nombre de las etiquetas (labels)

A continuación se extraen los distintos objetos de manera organizada, ubicando a cada objeto en su respectiva bolsa o cluster. Cada objeto tiene su respectivo Id que permite identificar cada frase particularmente. Luego, se observa la cantidad de objetos agrupados por cada bolsa.

#-----------------------------------------------------------------------------------------------------------------------------
# Organization of clusters



vCluster_0 = []

vCluster_1 = []

vCluster_2 = []

vCluster_3 = []

for index, label in enumerate(vLabels):

    if label == 0:

       vCluster_0.append(index)    

    if label == 1:

       vCluster_1.append(index) 

    if label == 2:

       vCluster_2.append(index)

    if label == 3:

       vCluster_3.append(index) 



print('\nClusters size')

print('Cluster 0: ', len(vCluster_0))

print('Cluster 1: ', len(vCluster_1))

print('Cluster 2: ', len(vCluster_2))

print('Cluster 3: ', len(vCluster_3))  



out []:

    - Clusters size
Cluster 0:  27092
Cluster 1:  9680
Cluster 2:  1086
Cluster 3:  3374

Para la recuperación de las frases AE una vez han sido clusterizadas, se extrae el ID correspondiente a cada objeto y se busca en la lista inicial del set de frases AE

#-----------------------------------------------------------------------------------------------------------------------------

# Recovery of sentences

vBag_0 = []

vBag_1 = []

vBag_2 = []

vBag_3 = []



for index, phrase in enumerate(vAnythingElse):

    if index in vCluster_0: 

       vBag_0.append(phrase)

    

    if index in vCluster_1: 

       vBag_1.append(phrase)

    

    if index in vCluster_2: 

       vBag_2.append(phrase)

    

    if index in vCluster_3: 

       vBag_3.append(phrase)

            
#-----------------------------------------------------------------------------------------------------------------------------
Presentación final de los datos: Almacenamiento de las frases agrupadas en un archivo Excel


# Save into a excel file



df0 = pd.DataFrame({'Bag 0':vBag_0})    

df1 = pd.DataFrame({'Bag 1':vBag_1})

df2 = pd.DataFrame({'Bag 2':vBag_2})

df3 = pd.DataFrame({'Bag 3':vBag_3})   

    

    

writer = pd.ExcelWriter('Anything_Else_Bag_of_words.xlsx')

df0.to_excel(writer,'Bag 0')

df1.to_excel(writer,'Bag 1')

df2.to_excel(writer,'Bag 2')

df3.to_excel(writer,'Bag 3')

writer.save()   

    
#-----------------------------------------------------------------------------------------------------------------------------
Evaluación del modelo

Al revisar los resultados obtenidos en esta primera fase, se encontró ambiguedades en la agrupación. De este modo la ruta de indagación tuvo que reorientarse. Con el propósito de determinar si se realizará un buen trabajo de predicción para nuevos y futuros datos es necesario evaluar el modelo elaborado. En este sentido se propone la elaboración de un Ground truth que proponga un punto de partida a una aproximación mas precisa de lo que se espera al agrupar.

En desarrollo...
