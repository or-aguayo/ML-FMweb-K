from random import random

from docker import client

import aprendizaje_automatico
import punto_variacion
import docker


class Mapek:
    def __init__(self):
        self._puntoVariacion = None

    def monitoreo(self, mc):
        numRandom = random.randint(1, 350)
        configuracion = aprendizaje_automatico.arbolesAleatoriosInverso("data/datos_redesneuronales.csv", numRandom)
        self.analizar(mc, configuracion)

    def analizar(self, mc, configuracion):
        self.conocimiento(configuracion, mc)
        self.planificar()

    def planificar(self):
        contenedores = self._puntoVariacion.obtenerConfiguracion()
        self.ejecutar(contenedores)

    def ejecutar(self, contenedores):
        for container in client.containers.list(all=True):
            if (container.name in contenedores):
                cont = client.containers.get(container.id)
                if (contenedores[container.name] == True and cont.status == "exited"):
                    cont.start()
                    print(f"contenedor {container.name} iniciado")
                elif (contenedores[container.name] == False and cont.status == "running"):
                    cont.stop()
                    print(f"contenedor {container.name} detenido")


    def conocimiento(self, configuracion, mc):
        self._puntoVariacion = punto_variacion.PuntoVariacion(configuracion, mc, "Gestor Aire")

    def getConocimiento(self):
        return self._puntoVariacion
