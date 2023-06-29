# TRABAJO FINAL DE MASTER
## Máster Universitario en Ciencia de Datos (Data Science)
### Estado de la computación cuántica en el aprendizaje por refuerzo y cómo aplicarla en DQN y Reinforce con Línea Base

El objetivo de este trabajo es el de analizar cómo se está aplicando la computación cuántica en problemas de aprendizaje automático con la finalidad de facilitar la adopción de esta nueva tecnología emergente.

La computación cuántica permitirá abarcar problemas más complejos y con mayor cantidad de variables que en computación clásica. Además, contribuirá al decremento de emisiones de gases de efecto invernadero, ya que reducirá el tiempo de computación de los entrenamientos.

Durante este Trabajo de Fin de Máster, se da una visión general de la computación cuántica y sus frameworks más relevantes, su aplicación en el ámbito del Deep Learning (DL) y se explora su implantación en algoritmos de Deep Reinforcement Learning (DQN y Reinforce con línea base). 

Se realiza una comparativa de los distintos algoritmos ejecutados en un computador clásico y un simulador cuántico. Se utiliza el framework Farama Foundation para entrenar y probar los agentes. Debido a las limitaciones a nivel de Qbits de los ordenadores cuánticos disponibles, se parte de un entorno de Gymnasium simple.

Palabras clave: Quantum Computing, Reinforcement Learning, DQN, Reinforce with Baseline

\subsection{Estructura del código}


Al llevar a cabo del desarrollo del proyecto, se han escogido como base las implementaciones propuestas por la UOC (https://github.com/jcasasr/Aprendizaje-por-refuerzo) en la asignatura "Aprendizaje por Refuerzo" del Máster Universitario en Ciencia de Datos para DQN (carpeta M09) y Reinforce con línea base (carpeta M10). De este modo, se facilita el aprendizaje por parte de otros estudiantes de esta asignatura que quieran profundizar en sus implementaciones cuánticas. Se ha reestructurado el código para agrupar código en funciones y simplificar su comprensión.

El código se compone de dos partes:
* Librerías definidas para la implementación de los QVCs, modelos, etc.
* Cuadernos Jupyter para lanzar los distintos entrenamientos para evaluar los resultados.
* Ejecuciones realizadas para los distintos escenarios considerados.

A continuación, se enumeran las diferentes librerías definidas:

* Agent.py: Contiene dos clases, una clase DQNAgent y una clase reinforceAgent que implementan los Agentes DQN y Reinforce con línea base respectivamente.
* Model.py: Contiene las clases DQN y PGReinforce que implementan los modelos a utilizar para DQN y Reinforce con línea base respectivamente.
* experienceReplayBuffer.py: En este módulo se implementa una clase experienceReplayBuffer para almacenar las experiencias pasadas del agente en el algoritmo DQN.
* QVC.py: En esta librería se implementa la clase QuantumNet donde se define el circuito cuántico variacional. 

