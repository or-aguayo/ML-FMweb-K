class CaracteristicaConfiguracion:
    def __init__(self, caracteristica, estado, contenedorDocker, href):
        self._caracteristica = caracteristica
        self._estado = estado
        self._contenedorDocker = contenedorDocker
        self._subCaracteristicas = []
        self._href = href

    @property
    def getCaracteristica(self):
        return self._caracteristica

    @property
    def getEstado(self):
        return self._estado

    @property
    def getContenedorDocker(self):
        return self._contenedorDocker

    @property
    def getSubcaracteristicas(self):
        return self._subCaracteristicas

    def agregarRelacion(self, caracteristica):
        self._subCaracteristicas.append(caracteristica)

    @property
    def getHref(self):
        return self._href




class PuntoVariacion:
    def __init__(self, estadoConfiguracion, grafoMC):
        self._modeloConfiguracion = self.agregarRelacionesCaracteristicas(self.agregarCaracteristicas(estadoConfiguracion),grafoMC)
        self._grafoCaracteristicas = grafoMC

    def obtenerConfiguracion(self):
        reconfiguracion = {}
        for caracteristica in self._modeloConfiguracion:
            reconfiguracion.update({caracteristica.getCaracteristica: caracteristica.getEstado})
        return reconfiguracion

    def obtenerConfiguracionNivel(self,nombreCaracteristica):
        reconfiguracion = {"links" : []}
        for caracteristica in self._modeloConfiguracion:
            if caracteristica.getCaracteristica == nombreCaracteristica:
                for subCaracteristica in caracteristica.getSubcaracteristicas:
                    if subCaracteristica.getEstado:
                        configuracion = {}
                        configuracion.update({"name" : subCaracteristica.getCaracteristica})
                        configuracion.update({"href": subCaracteristica.getHref})
                        reconfiguracion["links"].append(configuracion)

                return reconfiguracion
        return None
    def obtenerEstadoCaracteristica(self, nombreCaracteristicas):
        reconfiguracion = {}
        for caracteristica in self._modeloConfiguracion:
            if caracteristica.getCaracteristica == nombreCaracteristicas:
                configuracion = {}
                if(caracteristica.getEstado):
                    reconfiguracion.update({"name": caracteristica.getCaracteristica})
                    reconfiguracion.update({"href": caracteristica.getHref})
                return reconfiguracion
        return reconfiguracion


    def agregarCaracteristicas(self,estadoConfiguracion):
        modeloConfiguracion = []
        for caracteristica in estadoConfiguracion:
            estado = False
            if " activada" in caracteristica:
                estado = True
                caracteristica = caracteristica.replace(" activada", "")
            else:
                caracteristica = caracteristica.replace(" desactivada", "")
            href = "/"+caracteristica.replace(" ","_")
            caracteristicaConfiguracion = CaracteristicaConfiguracion(caracteristica,estado,caracteristica,href.lower())
            modeloConfiguracion.append(caracteristicaConfiguracion)
        return modeloConfiguracion

    def agregarRelacionesCaracteristicas(self, modeloConfiguracion, grafoMC):
        for caracteristicaConf in modeloConfiguracion:
            relacionesCaracteristica = grafoMC.obtenerRelacionesCaracteristica(caracteristicaConf.getCaracteristica)
            for relacion in relacionesCaracteristica:
                for subCaracteristicaConf in modeloConfiguracion:
                    if relacion == subCaracteristicaConf.getCaracteristica:
                        caracteristicaConf.agregarRelacion(subCaracteristicaConf)
        return modeloConfiguracion
