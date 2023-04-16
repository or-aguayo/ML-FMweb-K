import grafo_mc
import punto_variacion
import aprendizaje_automatico
def inicializar():
    #Generar el grafo para permutar todos los posibles estados del modelo
    mc = grafo_mc.generarPosiblesEstados()
    #Asociar una regla de adaptación para cada punto de variación creado en la permutación
    aprendizaje_automatico.guardarPredicciones(
        aprendizaje_automatico.redesNeuronales("data/dataset.csv", "data/datos.csv"),
        "data/datos_redesneuronales.csv")

    #Crear una regla adaptación, idealmente debe ser un numero random
    reglaAdaptacion1 = 250.2
    #Entrego un numero aleatorio random, que asimila ser una regla de adaptación, el cual a partir de lo anterior clasifica un punto de variación
    #Esta clasificación se realiza por medio del algoritmo de clasificación de arboles aleatorios
    puntoVariacion1 = aprendizaje_automatico.arbolesAleatoriosInverso("data/datos_redesneuronales.csv", reglaAdaptacion1)
    #presento el punto de variación 1
    print("Punto de variación 1, ", puntoVariacion1)

    objetoPuntoVariacion = punto_variacion.PuntoVariacion(puntoVariacion1,mc)
    print("JSON con configuracion ", objetoPuntoVariacion.obtenerConfiguracion())
    print("Caracteristicas activas del sub nivel Turismo ",objetoPuntoVariacion.obtenerConfiguracionNivel("Turismo"))
    print("Caracteristicas activas del sub nivel Entretenimiento ", objetoPuntoVariacion.obtenerConfiguracionNivel("Entretenimiento"))

inicializar()

