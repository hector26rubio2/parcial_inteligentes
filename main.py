from cargarDatos.cargarImagen import CargarImagen
from modelos.entrenamiento_1 import NeuralNetwork


class Main:

    def start_app(self):
        mode = int(input("1: Probar Modelo , 2: Entrenar modelo, 3: Llenar DataSet \n"))
        if mode == 1:
            pass
        if mode == 2:
            model = int(input("1: Primer modelo , 2: Segundo modelo, 3: Tercer modelo \n"))
            if model == 1:
                model = NeuralNetwork()
                model.crearModel(nombre="modeloMain")
            if model == 2:
                pass
            if model == 3:
                pass
        if mode == 3:
            carta = int(input("Ingrese el número de la carta: \n"))
            models= CargarImagen(carta=carta)
            models.run()



if __name__ == "__main__":
    main = Main()
    main.start_app()