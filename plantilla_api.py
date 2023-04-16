from fastapi import FastAPI
import grafo_mc
import aprendizaje_automatico
import json

app = FastAPI()

@app.on_event("startup")
async def iniciar_app():
    global mc
    global puntoVariacion
    mc = grafo_mc.generarPosiblesEstados()
    aprendizaje_automatico.guardarPredicciones(aprendizaje_automatico.redesNeuronales("data/dataset.csv", "data/datos.csv"),
                                               "data/datos_redesneuronales.csv")


@app.get("/obtenerReconfiguracion")
async def obtenerReconfiguracion(reglaAdaptacion : float):
    puntoVariacion = aprendizaje_automatico.arbolesAleatoriosInverso("data/datos_redesneuronales.csv", reglaAdaptacion)
    return puntoVariacion

@app.get("/obtenerReconfiguracionJSON")
async def obtenerReconfiguracionJSON(reglaAdaptacion : float):
    puntoVariacion = aprendizaje_automatico.obtenerJSONPrediccion(aprendizaje_automatico.arbolesAleatoriosInverso("data/datos_redesneuronales.csv", reglaAdaptacion))
    return puntoVariacion


#funcion que retorna el arbol completo de caracteristicas
@app.get("/obtenerArbol")
async def obtenerCaracteristicas():
    return mc.obtenerArbol()

#función que retorna las relaciones de una caracteristica
@app.get("/obtenerRelacionesCaracteristica")
async def obtenerRelacionesCaracteristica(caracteristica : str):
    return mc.obtenerRelacionesCaracteristica(caracteristica)

#función que retorna todas las relaciones del mc
@app.get("/obtenerRelacionesMC")
async def obtenerRelacionesMC():
    return mc.obtenerRelacionesMC()

