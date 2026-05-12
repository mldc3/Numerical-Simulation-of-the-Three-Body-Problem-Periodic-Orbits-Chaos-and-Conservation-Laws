'''
Diciembre de 2025
###############################################################################
CODIGO TRABAJO FINAL MODELIZACIÓN : EL PROBLEMA DE LOS 3 CUERPOS + SIST. SOLAR
###############################################################################
Autores: Lourdes Domínguez Cacho, Nerea Rico González, Pablo Martínez Rubio
4º Grado en Física 
Universidad de Alicante.
'''

#INTRODUCCIÓN #################################################################
'''
El código realizado se dedica a la resolución del problema de los 3 cuerpos con
distintas configuraciones de condiciones iniciales y diversos métodos numéricos.
También trataremos de resolver otros problemas gravitacionales de similar 
simetría, como el caso de plantear una versión siplificada del sist. solar.
'''
###############################################################################


#TABLA DE CONTENIDOS###########################################################
'''
1) Librerías
2) Formato de condiciones iniciales 
3) Config. y Definción de condiciones iniciales :
    -  Sol Lagrange            - Sol Skinny Pineapple
    -  Sol Broucke             - Sol Li Liao (caso particular)
    -  Sol Fig 8               - Sol en caída libre
    -  Sol Butterfly           - Sol caótica
    -  Sol Dragonfly           - sistema solar completo
    -  Sol Yarn                - sist binario con planetas
    - P3C 3D                   - sist tierra luna, sol
    - P3CR intercambio         - Ampliación: Nube Molecular , galaxias
    
4) Plotear condiciones iniciales 
5) Cálculo de la acelaración gravitatoria para los cuerpos
6) Métodos numéricos para la resolución del sistema
    - Método de Euler                   - Euler Bacward
    - Rungue Kutta Orden 5 (RKF45)      - Método de Verlet (Velocidades)
    
7) Resumen de las condiciones de simulación
8) Cálculo de errores (E y L )  y energías del sistema 
9) Funciones para graficas trayectorias y guardar GIFs
10) MAIN: Resolución de los distintos sistemas 
11) ANEXOS : CONFIGURACIONES ADICIONALES PARA LA PRESENTACIÓN
'''
###############################################################################


#1. LIBRERÍAS #################################################################
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import time
###############################################################################

#2.FORMATO DE CONDICIONES INICIALES ###########################################

class System:
    
    """
    Esta clase representa nuestro 'Universo', y es el formato que vamos a
    elegir para contener  toda la información necesaria para la simulación de 
    N cuerpos en nuestro sistema. Es el formulario en blanco que luego vamos
    a ir rellenando con los conjutnos de condiciones en cada caso.
    """
    
    def __init__(self, num_particulas: int, x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float):
        
        """
        Constructor: Aquí se ''prepara''  el formulario inicial del sistema.
        """
        # 1. Parámetros escalares (números)____________________________________
        self.num_particulas = num_particulas # Cantidad total de cuerpos (N)
        self.G = G  # Constante de gravitación universal G

        # 2. Propiedades físicas (Arrays de Numpy)_____________________________
        # x: Matriz de posiciones. Tamaño (N, 3), con  N filas (planetas) y
        #3 columnas (x, y, z), que son las posiciones (x,y,z).
        self.x = x        
        
        # v: Matriz de velocidades. Tamaño (N, 3) con N filas (1 por astro) y 
        # 3 columnas que son las componentes (vx,vy,vz)
        self.v = v        
        
        # m: Vector de masas. Tamaño (N,). Una lista con las masas 
        #de cada cuerpo
        self.m = m        
        #______________________________________________________________________

    def center_of_mass_correction(self):
        """
        Tendremos que ajustar el sistema para que el CM esté en (0,0,0) y no se
        vaya desplazando en la simulación hacia una dirección.
        """
        
        # PASO 1: Calcularemos  la Masa Total (M) del sistema definido_________
        # Para ello, sumaremos las masas de todas las partículas 
        M = np.sum(self.m)

        # PASO 2: Calcularemos la posición y velocidad del Centro de Masas_____
        # Para ello podemos recurrir a la funcion 'einsum' que lp que hace es
        #multiplicar masa * posición y sumar uno a uno en ambos arrays.
        # Es equivalente a hacer R_cm = (Sumatoria de m_i * r_i) / M
        x_cm = np.einsum("i,ij->j", self.m, self.x) / M  # Pos media ponderada
        v_cm = np.einsum("i,ij->j", self.m, self.v) / M  # Vel media ponderada

        # PASO 3: Corrección de las coordenadas del CM_________________________
        
        # Restamos la posición/velocidad del CM a cada partícula individual
        # como hacíamos en la dinámica molecular par que se nos quede centrado
        # y no se nos vaya desplazando.
        self.x -= x_cm
        self.v -= v_cm
        #______________________________________________________________________


###############################################################################

#3. CONFIG Y DEFINICIÓN DE CONDS. INICIALES DEL SISTEMA########################
    
    
'''
Ahora llegamos básicamente a la parte más importante del código para poder
definir los distintos sistemas que podríamos llegar a estudiar: Es el momento
de definir las distintas condiciones iniciales del sistema. Para ello, para no
tener que definir un montón de parámetros inicializándolos como siempre, vamos 
a tratar de realizar, basándonos en algunas referencias una especie de base
de datos con las condiciones inciales de los distintos astros y sistemas que 
nos gustaría estudiar con simulaciones.

Vamos a comenzar por lo más complicado, que es inicializar las condiciones 
iniciales del sistema solar, y así ya tenemos las constantes definidas,
posteriormente nos centaremos en los 3 cuerpos y en distintas soluciones del 
sistema y por último definiremos algunas configuraciones interesantes para e
estudiar dado el código realizado. Entre  ellas está el punto de ampliación
más importante : La realización de una versión simplificada del sistema solar 
con datos realistas


'''

def Cond_iniciales_definidas(cond_inicial):
    
    
    """
    Esta función es la que va a recoger todas las conndiciones iniciales que 
    vayamos a definir para nuestras simulaciones. Tú le das el nombre en texto 
    (cond_inicial) y ella busca los datos para configurar ese sist.
    
    Argumentos: 
        cond_inicial (str): Va a ser básicamente el nombre del sistema que
        quieres simular en formato de cadena de texto.
                                 
    
    Nos devuelve como returns:
        Una tupla con 4 elementos:
        1. system: El objeto System  de (posiciones, velocidades, masas).
        2. labels: Una lista con los nombres de los cuerpos para la gráfica
        3. colors: Una lista con los colores para pintar cada cuerpo.
        4. Un booleano (True/False) indicando si se debe mostrar la leyenda
          en un gráfico.
          
    FUENTEs: 
    Los datos numéricos de este código  para las masas vienen de :
                https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf
    Este artículo de the Astronomical Jorunal de 2021 básicamente recoge en el
    los valores de G*M medidos para distintos astros.
    
    Los datos de posiciones vienen de :
    Data dated on A.D. 2024-Jan-01 00:00:00.0000 TDB
    # Computational data generated by NASA JPL Horizons System 
                https://ssd.jpl.nasa.gov/horizons/
    """
    
    
    
    #Nuestra fuente nos da la información de los valores de GM en km^3/s^2,
    # sin embargo, estos valores no son nada comodos para visualizar nada, así
    #que algo que podríamos hacer para que sean más manejables sería pasarlos
    #de unidades a UA^3/dias^2. Así la escala es más apropiada.
    
    #--------------------------------------------------------------------------
    # PARTE 1:  MASAS DE LOS ASTROS DEL SISTEMA SOLAR Y CTE G
    # -------------------------------------------------------------------------

    # Vamos a definir nuestro factor de conversión de km^3 s^-2 a UA^3 d^-2
    CONVERSION_FACTOR = (86400**2) / (149597870.7**3)



    # Vamos a definir los valores de GM que cogemos del artículo antes de
    #convertirlos de unidades (km^3 s^-2)
    GM_Km_s = {
        "Sol"    : 132712440041.279419,   "Mercurio": 22031.868551,
        "Venus"  : 324858.592000,         "Tierra"  : 398600.435507,
        "Marte"  : 42828.375816,          "Júpiter" : 126712764.100000,
        "Saturno": 37940584.841800,       "Urano"   : 5794556.400000,
        "Neptuno": 6836527.100580,        "Luna"    : 4902.800118,
        "Plutón" : 975.500000,            "Ceres"   : 62.62890,
        "Vesta"  : 17.288245, 
        }

    # GM valores en (AU^3 d^-2)
    GM_UA_Dias = {
        "Sol"     : 132712440041.279419 * CONVERSION_FACTOR,
        "Mercurio": 22031.868551        * CONVERSION_FACTOR,
        "Venus"   : 324858.592000       * CONVERSION_FACTOR,
        "Tierra"  : 398600.435507       * CONVERSION_FACTOR,
        "Marte"   : 42828.375816        * CONVERSION_FACTOR,
        "Júpiter" : 126712764.100000    * CONVERSION_FACTOR,
        "Saturno" : 37940584.841800     * CONVERSION_FACTOR,
        "Urano"   : 5794556.400000      * CONVERSION_FACTOR,
        "Neptuno" : 6836527.100580      * CONVERSION_FACTOR,
        "Luna"    : 4902.800118         * CONVERSION_FACTOR,
        "Plutón"  : 975.500000          * CONVERSION_FACTOR,
        "Ceres"   : 62.62890            * CONVERSION_FACTOR,
        "Vesta"   : 17.288245           * CONVERSION_FACTOR,
    }
    
    
    
    #Transformemos tb las masas, para que sean más manejables todavía,en masas 
    #solares. Es decir, dividimos cada una de ellas entre el valor de GM de la
    #masa del sol.
    
    
    # MASAS DEL SISTEMA SOLAR EN FUNCION DE LA MASA DE SOL [M_sol^-1]
    MASAS_SIST_SOLAR = {
        "Sol": 1.0,
        "Mercurio": GM_Km_s["Mercurio"] / GM_Km_s["Sol"],
        "Venus"   : GM_Km_s["Venus"]    / GM_Km_s["Sol"],
        "Tierra"  : GM_Km_s["Tierra"]   / GM_Km_s["Sol"],
        "Marte"   : GM_Km_s["Marte"]    / GM_Km_s["Sol"],
        "Júpiter" : GM_Km_s["Júpiter"]  / GM_Km_s["Sol"],
        "Saturno" : GM_Km_s["Saturno"]  / GM_Km_s["Sol"],
        "Urano"   : GM_Km_s["Urano"]    / GM_Km_s["Sol"],
        "Neptuno" : GM_Km_s["Neptuno"]  / GM_Km_s["Sol"],
        "Luna"    : GM_Km_s["Luna"]     / GM_Km_s["Sol"],
        "Plutón"  : GM_Km_s["Plutón"]   / GM_Km_s["Sol"],
        "Ceres"   : GM_Km_s["Ceres"]    / GM_Km_s["Sol"],
        "Vesta"   : GM_Km_s["Vesta"]    / GM_Km_s["Sol"],
    }

    #Esto lo hace más sencillo porque así podemos en estas unidades donde la 
    #masa solar es 1 ya podemos tomar como G:
    G = GM_UA_Dias["Sol"]  #Así definnimos la G constnte grav.
 
    #--------------------------------------------------------------------------
    # PARTE 2:  POSICIONES Y VELOCIDADES DE LOS ASTROS DEL SISTEMA SOLAR 
    # -------------------------------------------------------------------------

    # Unidades : Distancia en UA y tiempo en Días
    # Posiciones (x, y, z) y Velocidades (vx, vy, vz) del Sistema Solar.
    # Fecha: 1 de Enero de 2024. Fuente: NASA JPL Horizons.
    
    #POSICIONES DE LOS ASRTROS DEL SISTEMA SOLAR (x,y,z)
    POS_SIST_SOLAR = {
        "Sol"     : [-7.967955691533730e-03, -2.906227441573178e-03, 2.103054301547123e-04],
        "Mercurio": [-2.825983269538632e-01, 1.974559795958082e-01, 4.177433558063677e-02 ],
        
        "Venus"   : [-7.232103701666379e-01, -7.948302026312400e-02, 4.042871428174315e-02],
        "Tierra"  : [-1.738192017257054e-01, 9.663245550235138e-01, 1.553901854897183e-04 ],
        
        "Marte"   : [-3.013262392582653e-01, -1.454029331393295e00, -2.300531433991428e-02],
        "Júpiter" : [3.485202469657674e00, 3.552136904413157e00, -9.271035442798399e-02   ],
        
        "Saturno" : [8.988104223143450e00, -3.719064854634689e00, -2.931937777323593e-01  ],
        "Urano"   : [1.226302417897505e01, 1.529738792480545e01, -1.020549026883563e-01   ],
        
        "Neptuno" : [2.983501460984741e01, -1.793812957956852e00, -6.506401132254588e-01  ],
        "Luna"    : [-1.762788124769829e-01, 9.674377513177153e-01, 3.236901585768862e-04 ],
        
        "Plutón"  : [1.720200478843485e01, -3.034155683573043e01, -1.729127607100611e00   ],
        "Ceres"   : [-1.103880510367569e00, -2.533340440444230e00, 1.220283937721780e-01  ],
        "Vesta"   : [-8.092549658731499e-02, 2.558381434460076e00, -6.695836142398572e-02 ],
    }
    
    #VELOCIDADES DE LOS ASTROS DEL SISTEMA SOLAR (vx,vy,vz)
    VEL_SIST_SOLAR = {
        "Sol"     : [4.875094764261564e-06, -7.057133213976680e-06, -4.573453713094512e-08 ],
        "Mercurio": [-2.232165900189702e-02, -2.157207103176252e-02, 2.855193410495743e-04 ],
        
        "Venus"   : [2.034068201002341e-03, -2.020828626592994e-02, -3.945639843855159e-04 ],
        "Tierra"  : [-1.723001232538228e-02, -2.967721342618870e-03, 6.382125383116755e-07 ],
        
        "Marte"   : [1.424832259345280e-02, -1.579236181580905e-03, -3.823722796161561e-04 ],
        "Júpiter" : [-5.470970658852281e-03, 5.642487338479145e-03, 9.896190602066252e-05  ],
        
        "Saturno" : [1.822013845554067e-03, 5.143470425888054e-03, -1.617235904887937e-04  ],
        "Urano"   : [-3.097615358317413e-03, 2.276781932345769e-03, 4.860433222241686e-05  ],
        
        "Neptuno" : [1.676536611817232e-04, 3.152098732861913e-03, -6.877501095688201e-05  ],
        "Luna"    : [-1.746667306153906e-02, -3.473438277358121e-03, -3.359028758606074e-05],
        
        "Plutón"  : [2.802810313667557e-03, 8.492056438614633e-04, -9.060790113327894e-04  ],
        "Ceres"   : [8.978653480111301e-03, -4.873256528198994e-03, -1.807162046049230e-03 ],
        "Vesta"   : [-1.017876585480054e-02, -5.452367109338154e-04, 1.255870551153315e-03 ],
    }
    #Vamos a darles colores asociados a cada uno para luego plotear
    COLORES_SIST_SOLAR = {
           "Sol": "orange"    , "Mercurio": "slategrey" , "Venus": "wheat", 
           "Tierra": "skyblue", "Marte": "red"          , "Júpiter": "darkgoldenrod", 
           "Saturno": "gold"  , "Urano": "paleturquoise", "Neptuno": "blue",           
           "Plutón": None     , "Ceres": None           , "Vesta": None,
    }

    #--------------------------------------------------------------------------
    # PARTE 3:  DEFINICIÓN DE LOS SITEMAS Y SIMULACIONES A REALIZAR
    # -------------------------------------------------------------------------
    
    '''
    Ahora, con todas nuestras constantes ya definidas, vamos a dar las
    configuraciones predefinidas que utilizaremos como configuraciones iniciales
    del sistema segun la cadena de texto que le pasemos a nuestra funcion
    '''
    #Veamos algunas opciones :
    
    #__________________________________________________________________________
    
    ##########CONFIGURACIONES PARA EL P3C RESTRINGIDO AL PLANO#################
    #           (FAMILIAS DE TRAYECTORIAS ESTABLES Y CAÓTICA)
    #__________________________________________________________________________
    
    #Procederemos análogamente al caso del sistema solar, pero más simple
    
    #1) SOL DE LAGRANGE(Triángulo Equilátero)__________________________________
    if  cond_inicial == "lagrange_3_body" :
        # Solución de Lagrange: triángulo equilátero
        v = 1.5
        R1 = np.array([10.0, 0.0, 0.0])
        R2 = np.array([-10.0, 0.0, 0.0])
        R3 = np.array([0.0, 17.3205, 0.0])
        V1 = np.array([v * math.cos(1.0472), v * math.sin(1.0472), 0.0])
        V2 = np.array([v * math.cos(1.0472), -v * math.sin(1.0472), 0.0])
        V3 = np.array([-v, 0.0, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=100)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #2) SOL DE BROUCKE_________________________________________________________
    elif cond_inicial == "broucke_3_body":
        # Solución de Broucke
        R1 = np.array([-0.9892620043, 0.0, 0.0])
        R2 = np.array([2.2096177241, 0.0, 0.0])
        R3 = np.array([-1.2203557197, 0.0, 0.0])
        V1 = np.array([0.0, 1.9169244185, 0.0])
        V2 = np.array([0.0, 0.1910268738, 0.0])
        V3 = np.array([0.0, -2.1079512924, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #3) SOL EN FIGURA DE 8_____________________________________________________
    elif cond_inicial == "figure_eight_3_body":
        # Solución infinita (Figura 8)
        R1 = np.array([0.97000436, -0.24308753, 0.0])
        R2 = np.array([-0.97000436, 0.24308753, 0.0])
        R3 = np.array([0.0, 0.0, 0.0])
        V1 = np.array([0.93240737 / 2, 0.86473146 / 2, 0.0])
        V2 = np.array([0.93240737 / 2, 0.86473146 / 2, 0.0])
        V3 = np.array([-0.93240737, -0.86473146, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    
    #4) SOLUCIÓN BUTTERFLY_____________________________________________________
    elif cond_inicial == "butterfly_3_body":
        # Milovan Šuvakov & Veljko Dmitrašinović - Butterfly 1
        R1 = np.array([-1.0, 0.0, 0.0])
        R2 = np.array([1.0, 0.0, 0.0])
        R3 = np.array([0.0, 0.0, 0.0])
        V1 = np.array([0.392955, 0.097579, 0.0])
        V2 = np.array([0.392955, 0.392955, 0.0])
        V3 = np.array([-0.78591, -0.195158, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particles=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #5) SOLUCIÓN DRAGONFLY_____________________________________________________
    elif cond_inicial == "dragonfly_3_body":
        # Dragonfly II.4.A
        R1 = np.array([-1.0, 0.0, 0.0])
        R2 = np.array([1.0, 0.0, 0.0])
        R3 = np.array([0.0, 0.0, 0.0])
        V1 = np.array([0.080584, 0.588836, 0.0])
        V2 = np.array([0.080584, 0.588836, 0.0])
        V3 = np.array([-0.161168, -1.177672, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
     #_________________________________________________________________________
    
    #6) SOLUCIÓN YARN__________________________________________________________
    elif cond_inicial == "yarn_3_body":
        # Yarn VI.2.A
        R1 = np.array([-1.0, 0.0, 0.0])
        R2 = np.array([1.0, 0.0, 0.0])
        R3 = np.array([0.0, 0.0, 0.0])
        V1 = np.array([0.464445, 0.39606, 0.0])
        V2 = np.array([0.464445, 0.39606, 0.0])
        V3 = np.array([-0.92889, -0.79212, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #7) SOL SKINNY PINEAPPLE___________________________________________________
    elif cond_inicial == "skinny_pineapple_3_body":
        # Skinny pineapple
        R1 = np.array([0.419698802831, 1.190466261252, 0.0])
        R2 = np.array([0.076399621771, 0.296331688995, 0.0])
        R3 = np.array([0.100310663856, -0.729358656127, 0.0])
        V1 = np.array([0.1022945660031, 0.687248445943, 0.0])
        V2 = np.array([0.148950262064, 0.240179781043, 0.0])
        V3 = np.array([-0.251244828060, -0.9274282269779, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #8) SOL LI_LIAO____________________________________________________________
    elif cond_inicial == "li_liao_3_body":
        # Xiaoming Li & Shijun Liao
        R1 = np.array([-1.0, 0.0, 0.0])
        R2 = np.array([1.0, 0.0, 0.0])
        R3 = np.array([0.0, 0.0, 0.0])
        V1 = np.array([0.2869236336, 0.0791847624, 0.0])
        V2 = np.array([0.2869236336, 0.0791847624, 0.0])
        V3 = np.array([-1.1476945344, -0.3167390496, 0.0])
        m = np.array([1.0, 1.0, 0.5])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #9) FREE FALL ( EN CAÍDA LIBRE)___________________________________________
    elif cond_inicial == "free_fall_3_body":
        # Caída libre
        R1 = np.array([-2.0, 0.0, 0.0])
        R2 = np.array([2.0, 0.0, 0.0])
        R3 = np.array([0.0, 3.46, 0.0])
        V1 = np.array([0.0, 0.0, 0.0])
        V2 = np.array([0.0, 0.0, 0.0])
        V3 = np.array([0.0, 0.0, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=0.5)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #10) PROBLEMA DE LOS 3 CUERPOS CAÓTICO_____________________________________
    elif cond_inicial == "chaotic_3_body":
        # Sistemas caóticos
        vel = 1.5
        R1 = np.array([10.0, 0.0, 0.0])
        R2 = np.array([20*math.cos(2.888903884)-10, 20*math.sin(2.888903884), 0.0])
        R3 = np.array([20*math.cos(2.888903884)-10, -20*math.sin(2.888903884), 0.0])
        V1 = np.array([0.0, vel, 0.0])
        V2 = np.array([2*vel, 0.0, 0.0])
        V3 = np.array([-2*vel, 0.0, 0.0])
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        system = System(num_particulas=3, x=x, v=v, m=m, G=100)
        system.center_of_mass_correction()
    
        labels = [None, None, None]
        colors = [None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    # 11) INTERCAMBIO CAÓTICO PROBLEMA DE LOS 3 CUERPOS RESTRINGIDO____________
    
    #Esta configuración es un test destinado a los anexos del trabajo, con ella se
    #busca hacer una breve referencia ilustrativa del problema restringido
    
    elif cond_inicial == "intercambio_caótico_P3CR":
        # Para simular el problema Restringido tneemos 
        #que tomar m3 despreciable
        m = np.array([1.0, 1.0, 0.001]) 
        
        # G=1 para simplificar el cálculo de velocidad orbital
        G_sim = 1.0 
        
        # DISTANCIA: Separamos los cuerpos 2 unidades (r=1 desde el centro)
        r = 1.0 
        
        # VELOCIDAD ORBITAL BINARIA (Para órbita circular estable)
        # Equilibrio fuerza centrípeta vs gravitatoria: v = sqrt(GM / 4r)
        v_12= np.sqrt(G_sim * 1.0 / (4 * r)) # v = 0.5
        
        # POSICIONES
        R1 = np.array([-r, 0.0, 0.0])   # 1 a la izquierda
        R2 = np.array([r, 0.0, 0.0])    # 2 a la derecha
        
        # El cuerpo 3 empieza cerca del 2, pero entre medias
        # Lo ponemos un poco desplazado para que "caiga" hacia la otra.
        R3 = np.array([0.1, 0.5, 0.0]) 
        
        # VELOCIDADES
        V1 = np.array([0.0, -v_12, 0.0])
        V2 = np.array([0.0, v_12, 0.0]) 
        
        # Velocidad del cuerpo pequeño:
        # Le damos una velocidad lateral para que no caiga directo,
        # sino que entre en órbita de evolución caótica.
        V3 = np.array([0.0, 0.01, 0.0]) 
        
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
        
        system = System(num_particulas=3, x=x, v=v, m=m, G=G_sim)
        system.center_of_mass_correction()
        
        labels = ["m1", "m2", "Viajero"]
        colors = ["red", "blue", "green"]
        legend = True
        
        return system, labels, colors, legend
    #__________________________________________________________________________
    #12) CASO GENERAL EN 3D_____________________________________________________
    elif cond_inicial == "random_3d_3_body":
        # Posiciones formando un triángulo en la base y uno arriba (Z distinto de 0)
        R1 = np.array([1.0, 0.0, -0.5])   # Abajo
        R2 = np.array([-1.0, 0.0, -0.5])  # Abajo
        R3 = np.array([0.0, 1.0, 0.5])    # Arriba (Z=0.5)
        
        # Velocidades con componentes en Z para que no caigan al plano inmediatamente
        V1 = np.array([0.2, 0.2, 0.1])
        V2 = np.array([-0.2, 0.2, -0.1])
        V3 = np.array([0.0, -0.4, 0.0])
        
        m = np.array([1.0, 1.0, 1.0])
    
        x = np.array([R1, R2, R3])
        v = np.array([V1, V2, V3])
    
        # G=1 para simplificar unidades teóricas
        system = System(num_particulas=3, x=x, v=v, m=m, G=1)
        system.center_of_mass_correction()
    
        labels = ["Cuerpo 1", "Cuerpo 2", "Cuerpo 3"]
        colors = ["red", "blue", "green"]
        legend = True
        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #__________________________________________________________________________
    
    ################  SISTEMA SOLAR AL COMPLETO Y OTRAS CONFIGS ###############
    #__________________________________________________________________________
    
    
    
    #1) SISTEMA SOLAR AL COMPLETO CON DATOS REALES_____________________________
    if cond_inicial == "solar_system_plus":
        m = np.array(
            [
                MASAS_SIST_SOLAR["Sol"]    , MASAS_SIST_SOLAR["Mercurio"], MASAS_SIST_SOLAR["Venus"],
                MASAS_SIST_SOLAR["Tierra"] , MASAS_SIST_SOLAR["Marte"]   , MASAS_SIST_SOLAR["Júpiter"],
                MASAS_SIST_SOLAR["Saturno"], MASAS_SIST_SOLAR["Urano"]   , MASAS_SIST_SOLAR["Neptuno"],
                MASAS_SIST_SOLAR["Plutón"] ,  MASAS_SIST_SOLAR["Ceres"]  , MASAS_SIST_SOLAR["Vesta"],
            ])
        
        #vamos a definir los elementos del array de posiciones como los datos:
        R1 = np.array(POS_SIST_SOLAR["Sol"])
        R2 = np.array(POS_SIST_SOLAR["Mercurio"])
        R3 = np.array(POS_SIST_SOLAR["Venus"])
        R4 = np.array(POS_SIST_SOLAR["Tierra"])
        R5 = np.array(POS_SIST_SOLAR["Marte"])
        R6 = np.array(POS_SIST_SOLAR["Júpiter"])
        R7 = np.array(POS_SIST_SOLAR["Saturno"])
        R8 = np.array(POS_SIST_SOLAR["Urano"])
        R9 = np.array(POS_SIST_SOLAR["Neptuno"])
        R10 = np.array(POS_SIST_SOLAR["Plutón"])
        R11 = np.array(POS_SIST_SOLAR["Ceres"])
        R12 = np.array(POS_SIST_SOLAR["Vesta"])
        
        #ahora las velocidades:
        V1 = np.array(VEL_SIST_SOLAR["Sol"])
        V2 = np.array(VEL_SIST_SOLAR["Mercurio"])
        V3 = np.array(VEL_SIST_SOLAR["Venus"])
        V4 = np.array(VEL_SIST_SOLAR["Tierra"])
        V5 = np.array(VEL_SIST_SOLAR["Marte"])
        V6 = np.array(VEL_SIST_SOLAR["Júpiter"])
        V7 = np.array(VEL_SIST_SOLAR["Saturno"])
        V8 = np.array(VEL_SIST_SOLAR["Urano"])
        V9 = np.array(VEL_SIST_SOLAR["Neptuno"])
        V10 = np.array(VEL_SIST_SOLAR["Plutón"])
        V11 = np.array(VEL_SIST_SOLAR["Ceres"])
        V12 = np.array(VEL_SIST_SOLAR["Vesta"])
        
        #Agrupo en arrays
        x = np.array([R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,])
        v = np.array([V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,])
        
        #rellenamos el formulario
        system = System(num_particulas=len(m),x=x,v=v,m=m, G=G,)
        #corregimos
        system.center_of_mass_correction()
        
        #generamos los colores y lo de la leyenda 
        labels = list(COLORES_SIST_SOLAR.keys())
        colors = list(COLORES_SIST_SOLAR.values())
        legend = True

        return system, labels, colors, legend
    #__________________________________________________________________________
    
    #2) SISTEMA DE ESTRELLA BINARIA CON PLANETAS_______________________________
    elif cond_inicial == "binary_star_4_body":
        R1 = np.array([15.0, 0.0, 0.0])
        R2 = np.array([-15.0, 0.0, 0.0])
        R3 = np.array([60.0, 0.0, 0.0])
        R4 = np.array([100.0, 0.0, 0.0])
        V1 = np.array([0.0, 10.0, 0.0])
        V2 = np.array([0.0, -10.0, 0.0])
        V3 = np.array([0.0, 20.0, 0.0])
        V4 = np.array([0.0, 15.0, 0.0])
        m = np.array([100.0, 100.0, 1/1000, 1/1000])
    
        x = np.array([R1, R2, R3, R4])
        v = np.array([V1, V2, V3, V4])
    
        system = System(num_particulas=4, x=x, v=v, m=m, G=100)
        system.center_of_mass_correction()
    
        labels = [None, None, None, None]
        colors = [None, None, None, None]
        legend = False
        return system, labels, colors, legend
    #__________________________________________________________________________
   
    #3) SOL TIERRA Y LUNA______________________________________________________
    elif cond_inicial == "earth_moon_sun_real":
        # Solo Sol, Tierra y Luna
        m = np.array([ MASAS_SIST_SOLAR["Sol"],MASAS_SIST_SOLAR["Tierra"],
            MASAS_SIST_SOLAR["Luna"]])
    
        x = np.array([POS_SIST_SOLAR["Sol"], POS_SIST_SOLAR["Tierra"],
            POS_SIST_SOLAR["Luna"],])
        v = np.array([VEL_SIST_SOLAR["Sol"], VEL_SIST_SOLAR["Tierra"],
            VEL_SIST_SOLAR["Luna"],])
        system = System(num_particulas=len(m), x=x,v=v,m=m,G=G,)
        system.center_of_mass_correction()
    
        labels = ["Sol", "Tierra", "Luna"]
        colors = ["orange", "skyblue", "grey"]
        legend = True

        return system, labels, colors, legend
    #__________________________________________________________________________



    #Estas son configuraciones pensadas para llevar el código al límite, y para
    #ello se ha recurrido a enlaces externos disponibles en la bibliografía para
    #conseguir inicializar condiciones iniciales y ver su evolución.

    #4) CASO NUBE MOLECULAR____________________________________________________
    elif cond_inicial ==  "plummer_cluster":
        # lee el archivo generado por plummer.py (kpc, km/s, 1e10 Msun)
        data = np.loadtxt("InitialConditions/PlummerIC.txt", comments="#")

        # ---- unidades originales de plummerIC ----
        # x,y,z en kpc
        # vx,vy,vz en km/s
        # masas en unidades de 1e10 Msun
        m_code = data[:, 0]      # [1e10 Msun]
        x_kpc  = data[:, 1:4]    # [kpc]
        v_kms  = data[:, 4:7]    # [km/s]
    
        # ---------- constantes físicas ----------
        kpc_to_m   = 3.08567758128e19      # 1 kpc = 3.08567758128e19 m
        km_to_m    = 1.0e3                 # 1 km = 1e3 m 
        Msun_to_kg = 1.98847e30            # 1 Msun ≈ 1.98847e30 kg 
    
        code_mass_to_kg = 1.0e10 * Msun_to_kg  # 1 unidad de masa del código = 1e10 Msun
    
        AU_to_m   = 1.495978707e11          # 1 AU en m 
        day_to_s  = 86400.0                 # 1 día en s
        G_SI      = 6.67430e-11             # m^3 kg^-1 s^-2 
    
        # ---------- convertir a SI ----------
        x_m   = x_kpc * kpc_to_m           # m
        v_m_s = v_kms * km_to_m            # m/s
        m_kg  = m_code * code_mass_to_kg   # kg
    
        # ---------- SI a AU, días, M_sol ----------
        x_AU     = x_m / AU_to_m
        v_AU_day = v_m_s * (day_to_s / AU_to_m)
        m_Msun   = m_kg / Msun_to_kg
    
        # G en unidades AU^3 d^-2 M_sun^-1
        G_code = G_SI * (day_to_s**2 / AU_to_m**3) * Msun_to_kg
    
        system = System(
            num_particulas=len(m_Msun),
            x=x_AU,
            v=v_AU_day,
            m=m_Msun,
            G=G_code,
        )
        system.center_of_mass_correction()
    
        labels = [None] * len(m_Msun)
        colors = ["white"] * len(m_Msun)
        legend = False
    
        return system, labels, colors, legend
    
    #5) VÍA LÁCTEA_____________________________________________________________
    elif cond_inicial == "milky_way":
        data = np.loadtxt("galaxyIC/InitialConditions/MilkyWayIC.txt", comments="#")
    
        m_code = data[:, 0]      # [1e10 Msun] 
        x_kpc  = data[:, 1:4]    # [kpc] 
        v_kms  = data[:, 4:7]    # [km/s] 
    
        # --- constantes ---
        kpc_to_m   = 3.08567758128e19
        km_to_m    = 1.0e3
        Msun_to_kg = 1.98847e30
        code_mass_to_kg = 1.0e10 * Msun_to_kg
    
        AU_to_m   = 1.495978707e11
        day_to_s  = 86400.0
        G_SI      = 6.67430e-11
    
        # ---------- convertir a SI ----------
        x_m   = x_kpc * kpc_to_m
        v_m_s = v_kms * km_to_m
        m_kg  = m_code * code_mass_to_kg
    
        # ---------- SI a AU, días, M_sol ----------
        x_AU     = x_m / AU_to_m
        v_AU_day = v_m_s * (day_to_s / AU_to_m)
        m_Msun   = m_kg / Msun_to_kg
    
        G_code = G_SI * (day_to_s**2 / AU_to_m**3) * Msun_to_kg
    
        system = System(num_particulas=len(m_Msun), x=x_AU, v=v_AU_day, m=m_Msun, G=G_code)
        system.center_of_mass_correction()

        labels = [None] * len(m_Msun)
        colors = ["gray"] * len(m_Msun)
        legend = False
    
        return system, labels, colors, legend

    #6) GALAXIA ESPIRAL________________________________________________________
    elif cond_inicial == "soles_espiral":
        data = np.loadtxt("soles_espiral_real.txt", comments=None, skiprows=1)
        
        x_norm = data[:, 0:3]; v_norm = data[:, 3:6]; m_norm = data[:, 6]
        
        AU_por_pixel = 0.25      # Escala para brazos largos
        dt_nbody = 0.1
        day_to_s = 86400.0
        AU_to_m = 1.495978707e11
        Msun_to_kg = 1.98847e30
        G_SI = 6.67430e-11
        G_nbody = 1e2
    
        x_AU = x_norm * AU_por_pixel
        v_AU_day = v_norm * (AU_por_pixel / dt_nbody) * (day_to_s / AU_to_m)
        m_Msun = m_norm * 1.0
    
        G_code = G_SI * (day_to_s**2 / AU_to_m**3) * Msun_to_kg / G_nbody
    
        system = System(num_particulas=len(m_Msun), x=x_AU, v=v_AU_day, m=m_Msun, G=G_code)
        system.center_of_mass_correction()
    
        colors = ["black"] + ["gold"] * 1080  # Agujero negro + estrellas doradas
        labels = [None] * len(m_Msun)
        legend = False
    
        return system, labels, colors, legend
#______________________________________________________________________________

    # Por si acaso hay algun error
    else:
        raise ValueError(f"ILa cond incial introducida no se ha reconocido: {cond_inicial}.")

###############################################################################
###############################################################################



#4. PLOTEAR CONDICIONES INICIALES #############################################

"""
Esta sección es prescindible pero sirve para comprobar si las condiciones
iniciales introducidas son razonables.
"""


def plot_initial_conditions(system: System,labels: list,colors: list,legend: bool, title: str = None) -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("$x$ (AU)")
    ax.set_ylabel("$y$ (AU)")
    ax.set_title(title)

    for i in range(system.num_particulas):
        ax.scatter(system.x[i, 0], system.x[i, 1], marker="o", color=colors[i], label=labels[i])

    if legend:
        ax.legend()

    plt.show()

###############################################################################
###############################################################################



#5 FUNCIÓN DE CÁLCULO DE LA ACELERACIÓN GRAVITATORIA PARA LOS CUERPOS##########

#Ahora vamos a otro punto importante.Como ya contamos con las condiciones 
#iniciales, ahora es momento de hallar las aceleraciones de unos cuerpos debido
# a la interacción gravitatoria entre ellos. Vamos a tratar de vectorizar la 
#expresión de calculo de la aceleración.

def calcula_a(a, system): 
    # Extrae las posiciones, masas y constante gravitacional del sistema que 
    #definimos con las condiciones iniciales
    x = system.x   # Pos de los cuerpos, forma (N, 3) : (x,y,z)
    m = system.m   # Masas de los cuerpos
    G = system.G   # Constante gravitacional

    # Vamos a hallar las r_ij relativas entre cada uno de nuestros cuerpos del
    # sistema con un formato vectorizado tipo (N,N,3):
    r_ij = x[:, np.newaxis, :] - x[np.newaxis, :, :]

    # Hallamos las r de las distanicas relativas
    #r_norm[i,j] = ||x[j] - x[i]||, forma (N, N)
    r_norm = np.linalg.norm(r_ij, axis=2)

    # Calculamos 1 / r^3 elemento por elemento. Hay que lidiar con dividir por0,
    #como no contamos autointeracciones vamos a poner un 0 para la diagonal.
    inv_r_cubo = np.divide(1.0, r_norm**3, out=np.zeros_like(r_norm), where=r_norm!=0)

    # Calculemos la aceleración resultante de cada partícula como la suma de a.
    #La vamos a hallar como : G* sumatorio(m*r_ij/r^3)) con m de cada cuerpo.
    
    a[:] = G * np.sum(r_ij * inv_r_cubo[:, :, np.newaxis] * m[:, np.newaxis, np.newaxis], axis=0)
    
    
###############################################################################



#6. MÉTODOS NUMÉRICOS PARA LA RESOLUCIÓN DLE SISTEMA###########################


#6.1) MÉTODO DE EULER _________________________________________________________
def paso_Euler(a, system, dt):
    calcula_a(a, system)     # Calcula a de las partículas en pos actual
    system.x += system.v * dt   # Actualiza pos con v actual (x_i(t+dt) = x_i(t) + v_i*dt)
    system.v += a * dt          # Actualiza v usando a calculada (v_i(t+dt) = v_i(t) + a_i*dt)
#______________________________________________________________________________

    
#6.2) METODO DE RUNGUE-KUTTA A ORDEN 5_________________________________________
def paso_rk5(a, system, dt):
    etapas = 6  # k1 a k6

    # Coeficientes de Dormand-Prince para las etapas intermedias
    #c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    a_mat = np.zeros((6, 5))
    a_mat[1,0] = 1/4
    a_mat[2,0:2] = [3/32, 9/32]
    a_mat[3,0:3] = [1932/2197, -7200/2197, 7296/2197]
    a_mat[4,0:4] = [439/216, -8, 3680/513, -845/4104]
    a_mat[5,0:5] = [-8/27, 2, -3544/2565, 1859/4104, -11/40]

    # Pesos finales para la combinación (orden 5)
    b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    # Guardamos posiciones y velocidades iniciales
    x0 = system.x.copy()
    v0 = system.v.copy()

    # Arrays para incrementos
    xk = np.zeros((etapas, system.num_particulas, 3))
    vk = np.zeros((etapas, system.num_particulas, 3))

    # Etapa 1 (k1)
    calcula_a(a, system)
    xk[0] = v0
    vk[0] = a

    # Etapas k2 a k6
    for i in range(1, etapas):
        system.x = x0.copy()
        system.v = v0.copy()
        for j in range(i):
            system.x += dt * a_mat[i,j] * xk[j]
            system.v += dt * a_mat[i,j] * vk[j]
        calcula_a(a, system)
        xk[i] = system.v
        vk[i] = a

    # Combinación ponderada de todas las etapas
    dx = np.zeros_like(system.x)
    dv = np.zeros_like(system.v)
    for i in range(etapas):
        dx += b[i] * xk[i]
        dv += b[i] * vk[i]

    # Actualizamos finalmante la posición y velocidad
    system.x = x0 + dt * dx
    system.v = v0 + dt * dv
#______________________________________________________________________________

#6.3) BACKWARD EULER___________________________________________________________
def backward_euler(a: np.ndarray, system: System, dt: float, max_iter: int = 10, tol: float = 1e-8) -> None:
    # Guardamos las posiciones y velocidades actuales
    x_old = system.x.copy()
    v_old = system.v.copy()
    
    # Inicializamos la predicción de la nueva posición con Euler explícito
    x_new = system.x + dt * system.v
    
    for iteration in range(max_iter):
        system.x = x_new
        calcula_a(a, system)         # a(x_new)
        v_new = v_old + dt * a       # v(t+dt)
        x_next = x_old + dt * v_new  # ecuación ackward Euler

        # Comprobar convergencia 
        if np.linalg.norm(x_next - x_new) < tol:
            x_new = x_next
            break

        x_new = x_next

    # Actualizamos el sistema con las nuevas posiciones y velocidades
    system.x = x_new
    system.v = v_new
    
#______________________________________________________________________________

#6.4) VERLET DE VELOCIDADES____________________________________________________
def paso_verlet_velocidades(a, system, dt):
    
    calcula_a(a, system)       # a(t)
    a_old = a.copy()           # guardamos a(t)

    # Paso de posición
    system.x += system.v * dt + 0.5 * a_old * dt**2  # x(t+dt)

    # Calculamos la aceleración en la nueva posición
    calcula_a(a, system)       # a(t+dt)

    # Paso de velocidad usando promedio de aceleraciones
    system.v += 0.5 * (a_old + a) * dt               # v(t+dt)
#______________________________________________________________________________

###############################################################################


#7. RESUMEN DE LAS CONDS DE SIMULACIÓN#########################################
def print_resumen_simulacion( system,tf,dt,num_pasos,paso_guardado,tamaño_sol):
    print("----------------------------------------------------------")
    print("Resumen de la simulación:")
    print(f"Nº cuerpos: {system.num_particulas}")
    print(f"G: {system.G}")
    print(f"tf: {tf} dias (Actual tf = dt * num_pasos = {dt * num_pasos} dias)")
    print(f"dt: {dt} dias")
    print(f"Nº pasos: {num_pasos}")
    print()
    print(f"Paso guardado para plot: {paso_guardado} dias")
    print(f"Tamaño solucion: {tamaño_sol}")
    print("----------------------------------------------------------")
###############################################################################


#8. CÁLCULO DE ERRORES Y ESTUDIO DE LAS ENERGÍAS DEL SISTEMA###################

#8.1) ERRORES EN LAS ENERGÍAS__________________________________________________
# Hallemos ell error en las energías de la simulación
def compute_rel_energy_error(sol_x, sol_v, system):
    n_pasos = sol_x.shape[0]              # Número de pasos temporales
    num_particulas = system.num_particulas # Número de partículas
    m = system.m                          # Vector de masas
    G = system.G                          # Constante gravitatoria
    rel_E_error = np.zeros(n_pasos)  # Array para almacenar Etotal en cada paso

    
    for n in range(n_pasos): 
        x = sol_x[n]    # Posiciones de todas las partículas en paso n
        v = sol_v[n]    # Velocidades de todas las partículas en paso n
        for i in range(num_particulas):
            rel_E_error[n] += 0.5 * m[i] * np.linalg.norm(v[i]) ** 2
            for j in range(i + 1, num_particulas):
                rel_E_error[n] -= G * m[i] * m[j] / np.linalg.norm(x[i]- x[j])


    E_inicial = rel_E_error[0] # E inicial
    rel_energy_error = (rel_E_error - E_inicial) / E_inicial  # Error relativo
    rel_energy_error = np.abs(rel_energy_error)               # Valor absoluto
   
    return rel_energy_error # Devuelve array con el error rel de cada paso


#Los graficaremos usando esto :
def plot_rel_E_error(rel_E_error, sol_t  ,title= None):
    plt.figure()
    plt.title(title)
    plt.plot(sol_t, rel_E_error)  # Dibuja el error relativo frente al tiempo
    plt.yscale("log")             # Escala logarítmica en el eje y
    plt.xlabel("Paso temporal")
    plt.ylabel("Error relativo en E")         
    plt.title("Error relativo en E vs Paso Temporal")   
    plt.show()   
    
#8.2 HALLAR ERRORES EN EL CÁLCULO DE MOMENTOS ANGULARES DEL SIST_______________

def compute_rel_angular_momentum_error(sol_x, sol_v, system):
    n_pasos = sol_x.shape[0]                   # Número de pasos temporales
    num_particulas = system.num_particulas     # Número de partículas
    m = system.m                               # Vector de masas

    L_total = np.zeros((n_pasos, 3))           # Momento angular total en cada paso

    for n in range(n_pasos):
        for i in range(num_particulas):
            r = sol_x[n, i]
            v = sol_v[n, i]
            L_total[n] += m[i] * np.cross(r, v)

    L0 = np.linalg.norm(L_total[0])             # Módulo del momento angular inicial
    rel_L_error = np.linalg.norm(L_total - L_total[0], axis=1) / L0
    rel_L_error = np.abs(rel_L_error)

    return rel_L_error

def plot_rel_L_error(rel_L_error, sol_t, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.plot(sol_t, rel_L_error)
    plt.yscale("log")
    plt.xlabel("Tiempo")
    plt.ylabel("Error relativo en |L|")
    plt.title("Error relativo del momento angular vs tiempo")
    plt.grid(True)
    plt.show()

#8.3 ESTUDIO DE LAS ENERGÍAS DE CADA CUERPO Y ENERGÍA TOTAL :__________________

def calcular_energias_individuales(sol_x, sol_v, system):
    n_pasos = sol_x.shape[0]
    num_particulas = system.num_particulas
    m = system.m
    G = system.G
    
    # Ahora las dimensiones de los array de cada tipo son (Pasos, Cuerpos).
    #Cada array recoge lo de cada particula en cada uno de los pasos.
    T_ind = np.zeros((n_pasos, num_particulas)) 
    V_ind = np.zeros((n_pasos, num_particulas)) 
    E_ind = np.zeros((n_pasos, num_particulas)) 
    

    for n in range(n_pasos):
        x_step = sol_x[n]
        v_step = sol_v[n]
        
        #calculamos la energía cinética de cada particula : 
        for i in range(num_particulas):
            # Energía Cinética Individual (T_i) : 
            v_sq = np.sum(v_step[i]**2)
            T_ind[n, i] = 0.5 * m[i] * v_sq
            
            # Energía Potencial Individual (V_i) :
            # Sumamos la interacción de este cuerpo 'i' con todos los demás 'j'
            potencial_i = 0.0
            for j in range(num_particulas):
                if i != j:
                    dist = np.linalg.norm(x_step[i] - x_step[j])
                    potencial_i -= G * m[i] * m[j] / np.sqrt(dist**2)
            
            V_ind[n, i] = potencial_i
            
            # Energía Total Individual 
            # Nota: La suma de estas E_ind NO es la E_total del sistema 
            # (porque contarías la potencial dos veces), pero sirve para ver quién escapa.
            E_ind[n, i] = T_ind[n, i] + V_ind[n, i]
            
    return T_ind, V_ind, E_ind



#Ahora que ya tenemos la energía de cada cuerpo, lo que vamos a hacer es 
#plotearlas, y para ello recurrimos a esta función:
    
def plot_energias_con_total(E_ind, E_sistema, t, labels, colors):
    
    plt.figure(figsize=(10, 6))
    plt.title("Energías Individuales vs Energía Total del Sistema")
    
    # 1. Plot de las energías individuales
    for i in range(E_ind.shape[1]):
        plt.plot(t, E_ind[:, i], label=f"E_ind {labels[i]}", color=colors[i], alpha=0.6, linewidth=1)
    
    # 2. Plot de la SUMA TOTAL DEL SISTEMA: (Esto es solo para ver que se conserva)
    plt.plot(t, E_sistema, label="TOTAL SISTEMA (Suma)", color="black", linewidth=2.5, linestyle="-")
    
    # Línea de referencia 0 para que sea un poco mas visual 
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel("Tiempo")
    plt.ylabel("Energía")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

#______________________________________________________________________________
###############################################################################


#9. GRAFICAR LAS TRAYECTORIAS   Y GENERACION DE GIFS###########################

#9.1) GRAFICAR LA TRAYECTORIA__________________________________________________

def plot_trayectoria(sol_x, labels, colors,legend,title = None):
    fig = plt.figure()  
    ax = fig.add_subplot(111, aspect="equal")  #subplot con aspecto igual (escala x=y)
    ax.set_xlabel("$x$ (AU)")
    ax.set_ylabel("$y$ (AU)")
    ax.set_title(title)
    # Itera sobre cada cuerpo y plotea la trayectoria 2D  usando
    #el color correspondiente.
    for i in range(sol_x.shape[1]):                       
        traj = ax.plot(sol_x[:, i, 0],sol_x[:, i, 1],color=colors[i],)
        ax.scatter(sol_x[-1, i, 0],sol_x[-1, i, 1],
        marker="o", color=traj[0].get_color(),label=labels[i],)  
        # se marca la posición final de la partícula

    if legend:   # Comprobamos si se debe mostrar la leyenda.
        fig.legend(loc="center right", borderaxespad=0.2) # Muestra la leyenda con las etiquetas de las partículas.
        fig.tight_layout()  
    # Ajusta automáticamente la figura para que los elementos no se sobrepongan.            

    plt.show()
    
"""
#la siguiente función es lo mismo que arriba pero la utilizamos solo para
#el apartado de ampliación de la galaxia.

def plot_trayectoria(sol_x, labels, colors, legend, title=None, margin=0.1):

    #Plotea la trayectoria 2D de las partículas usando las posiciones iniciales
    #para fijar los ejes y añadir un margen.
    
    #sol_x: array de posiciones (N_pasos, N_cuerpos, 3)
    #labels: lista de etiquetas de partículas
    #colors: lista de colores de partículas
    #legend: bool, si mostrar leyenda
    #margin: porcentaje de margen adicional sobre el rango inicial (0.1 = 10%)

    fig = plt.figure()  
    ax = fig.add_subplot(111, aspect="equal")  # subplot con aspecto igual (escala x=y)
    ax.set_xlabel("$x$ (AU)")
    ax.set_ylabel("$y$ (AU)")
    ax.set_title(title)

    # Ajustar ejes usando posiciones iniciales
    x0 = sol_x[0,:,0]
    y0 = sol_x[0,:,1]
    x_min, x_max = x0.min(), x0.max()
    y_min, y_max = y0.min(), y0.max()
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    R = 0.5 * max(x_max - x_min, y_max - y_min) * (1 + margin)
    ax.set_xlim(x_mid - R, x_mid + R)
    ax.set_ylim(y_mid - R, y_mid + R)

    # Itera sobre cada cuerpo y dibuja la trayectoria
    for i in range(sol_x.shape[1]):                       
        traj = ax.plot(sol_x[:, i, 0], sol_x[:, i, 1], color=colors[i])
        ax.scatter(sol_x[-1, i, 0], sol_x[-1, i, 1],
                   marker="o", color=traj[0].get_color(), label=labels[i])

    if legend:   # Mostrar leyenda si corresponde
        fig.legend(loc="center right", borderaxespad=0.2)
        fig.tight_layout()            

    plt.show()
"""

#______________________________________________________________________________

#9.2) HACER EL GIF DE LA TRAYECTORIA EN 3D_____________________________________
def hacer_gif_trayectoria_3D(sol_x, labels, colors,title, filename="orbitas_3D.gif", step=1): 
    
    # Definimos las dimensiones :
    N_pasos, N_cuerpos, _ = sol_x.shape
    frames = range(0, N_pasos, max(1, int(step))) # Lista de frames a pintar
    print("frames:", frames)

    #Configuramos la  figura 3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x (UA)")
    ax.set_ylabel("y (UA)")
    ax.set_zlabel("z (UA)")
    ax.set_title (f"{title}")

    #Creamos líneas y puntos vacíos (objetos a animar)
    # traj: las líneas de la trayectoria completa
    traj = [ax.plot([], [], [], color=colors[i], label=labels[i])[0] for i in range(N_cuerpos)]
    # points: la bolita que marca la posición actual
    points = [ax.plot([], [], [], "o", color=colors[i])[0] for i in range(N_cuerpos)]
    
    ax.legend() # Leyenda automática

    #Ajustamos límites fijos (IMPORTANTE para que no 'baile' la cámara)
    # Cogemos el mínimo y máximo global de TODA la simulación para fijar la caja
    ax.set_xlim(sol_x[:,:,0].min(), sol_x[:,:,0].max())
    ax.set_ylim(sol_x[:,:,1].min(), sol_x[:,:,1].max())
    ax.set_zlim(sol_x[:,:,2].min(), sol_x[:,:,2].max())

    print("Comenzamos función que llama a los fotogramas...")
    #Función que se llama en cada fotograma
    def update(frame):
        for i in range(N_cuerpos):
            # Pinta la trayectoria desde el inicio (0) hasta el frame actual
            # Matplotlib 3D requiere separar X,Y (set_data) de Z (set_3d_properties)
            """traj[i].set_data(sol_x[:frame, i, 0], sol_x[:frame, i, 1])
            traj[i].set_3d_properties(sol_x[:frame, i, 2])"""
            traj[i].set_data(sol_x[:frame, i, 0], sol_x[:frame, i, 1])
            traj[i].set_3d_properties(sol_x[:frame, i, 2])
            
            # Pinta el punto exacto en el frame actual
            """points[i].set_data([sol_x[frame, i, 0]], [sol_x[frame, i, 1]])
            points[i].set_3d_properties([sol_x[frame, i, 2]])"""
            points[i].set_data([sol_x[frame, i, 0]], [sol_x[frame, i, 1]])
            points[i].set_3d_properties([sol_x[frame, i, 2]])
        if i % (NUM_STEPS//10) == 0:
            print(f"Simulación al {step*100//NUM_STEPS}% completada...")
        return traj + points

    #Crear y guardar animación
    print(f"Generando GIF '{filename}'...")
    # interval=30 son ms entre frames (aprox 30 fps)
    anim = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
    
    # Usamos PillowWriter que viene por defecto para GIFs
    anim.save(filename, writer=PillowWriter(fps=30))
    
    plt.close(fig) # Cerramos para liberar memoria
    print("¡GIF guardado!")
    return anim
#______________________________________________________________________________
    
#9.3) Setear los ejes en 3D____________________________________________________

def set_3d_axes_equal(ax: plt.Axes):

    x_limits = ax.get_xlim3d()  
    y_limits = ax.get_ylim3d()  
    z_limits = ax.get_zlim3d()  #

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])  
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])  
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])  

#______________________________________________________________________________

#9.4)HACER EL GIF DE LA TRAYECTORIA EN EL PLANO XY_____________________________
def make_gif_trajectory(sol_x, labels, colors, filename="orbitas.gif", title=None, step=150000, fps=1):
    
    N_pasos, N_particulas, _ = sol_x.shape
    Pasos = range(0, N_pasos, step)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (AU)")
    ax.set_ylabel("$y$ (AU)")
    ax.grid(True)

    if title:
        ax.set_title(title)

    # Inicializamos las trayectorias y puntos
    trajectories = [ax.plot([], [], color=colors[i], label=labels[i])[0] for i in range(N_particulas)]
    points = [ax.plot([], [], "o", color=colors[i])[0] for i in range(N_particulas)]

    if labels:
        ax.legend(fontsize=6)

    xs = sol_x[:, :, 0]
    ys = sol_x[:, :, 1]
    margin = 0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    print("Pasos: ", Pasos)
    def update(frame_idx):
        frame = Pasos[frame_idx]
        # Imprimimos progreso cada 100 pasos
        if frame % 1000 == 0:
            print(f"Procesando paso {frame} / {N_pasos}")

        for i in range(N_particulas):
            trajectories[i].set_data(sol_x[:frame+1, i, 0], sol_x[:frame+1, i, 1])
            points[i].set_data([sol_x[frame, i, 0]], [sol_x[frame, i, 1]])

        return trajectories + points

    anim = FuncAnimation(fig, update, frames=len(Pasos), interval=0.01, blit= False )  # 1ms por frame
    anim.save(filename, writer=PillowWriter(fps=fps))
     
    plt.close(fig)
    print(f"GIF guardado como: {filename}")
    
    return anim
#______________________________________________________________________________
###############################################################################


#10. MAIN: LLAMADA A LAS FUNCIONES Y SIMULACIÓN ###############################


#A continuación estan toda nuestras definciones de parámetros de simulación
#segun el caso con el que vamos a trabajar

####################INICIALIZACIÓN DE CONDS PROB 3 CUERPOS#####################

#PROBLEMA DE LOS 3 CUERPOS CON LAGRANGE________________________________________
"""
TF = 40                  # Tiempo final de simulación en segundos
DT = 0.0001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración
sistemas= ["lagrange_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS BROUCKE_______________________________________________________
"""
TF = 20                  # Tiempo final de simulación en segundos
DT = 0.00001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración
sistemas= ["broucke_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS (FIG 8)_______________________________________________________
"""
TF = 10                  # Tiempo final de simulación en segundos
DT = 0.00002              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración
sistemas= ["figure_eight_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS BUTTERFLY_____________________________________________________
""" #no he conseguido que salga bien la trayectoria
TF = 10                  # Tiempo final de simulación en segundos
DT = 0.0000001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración
sistemas= ["butterfly_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS DRAGONFLY_____________________________________________________
"""
TF = 30                  # Tiempo final de simulación en segundos
DT = 0.000001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["dragonfly_3_body"]
"""
#______________________________________________________________________________
# PROB DE 3 CUERPOS YARN_______________________________________________________
"""
TF = 20                   # Tiempo final de simulación en segundos
DT = 0.00001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1       # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT)  # Número total de pasos de integración

sistemas = ["yarn_3_body"] 
"""
#______________________________________________________________________________
# PROB 3 CUERPOS SKINNY PINEAPPLE______________________________________________
"""

TF = 12                 # Tiempo final de simulación en segundos
DT = 0.000005              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["skinny_pineapple_3_body"]
"""
#______________________________________________________________________________
#PROB 3 CUERPOS LI LIAO________________________________________________________
"""
TF = 6                  # Tiempo final de simulación en segundos
DT = 0.000001              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración


sistemas = ["li_liao_3_body"]
"""
#______________________________________________________________________________
#PROB 3 CUERPOS CAÍDA LIBRE____________________________________________________
"""
TF = 30 #60                 # Tiempo final de simulación en segundos
DT = 0.0002              # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["free_fall_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS CAÓTICO_______________________________________________________
"""
TF = 200                # Tiempo final de simulación en segundos
DT = 0.008           # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["chaotic_3_body"]
"""
#______________________________________________________________________________
# PROB 3 CUERPOS CAÓTICO RESTRINGIDO___________________________________________
'''
#Esta configuración es un test destinado a los anexos del trabajo, con ella se
#busca hacer una breve referencia ilustrativa del problema restringido
TF = 25            # Tiempo suficiente para algun cruce
DT = 0.0005           # Paso de tiempo razonable para G=1
OUTPUT_INTERVAL = 1 
NUM_STEPS = int(TF / DT)
sistemas = ['intercambio_caótico_P3CR']
'''
#______________________________________________________________________________

#############################EXTRAS############################################

# SISTEMA SOLAR COMPLETO // SISTEMA TIERRA-LUNA-SOL____________________________
"""
TF = 1 * 365.24               # Tiempo final de simulación en días (200 años)
DT = 0.5                        # Paso de tiempo en días
OUTPUT_INTERVAL = 0.5 * 365.24  # Intervalo de salida de resultados en días (0.1 años)
NUM_STEPS = int(TF / DT)        # Número total de pasos de integración

# Lista de sistemas que quieres simular con esas condiciones iniciales
#sistemas= ["solar_system_plus", "earth_moon_sun_real"]
sistemas= ["earth_moon_sun_real"]
"""
#______________________________________________________________________________
# SISTEMA BINARIO CON 4 CUERPOS________________________________________________
"""
TF = 40      #200            # Tiempo final de simulación en segundos
DT = 0.000001     #0.02          # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["binary_star_4_body"]
"""
#______________________________________________________________________________

'''
Estas distribuciones de ampliación son las pensadas para llevar al código al 
extremo, son versiones simplificadas de galaxias y nubes moleculares que
requieren elevado tiempo de ejecución.
'''
# VÍA LÁCTEA __________________________________________________________________
'''
TF = 1.735e9   
TF = 5e5*100000  
DT = 5e5  
NUM_STEPS = int(TF / DT)   

OUTPUT_INTERVAL = 1
sistemas = ["milky_way"]
'''
#______________________________________________________________________________
# COND INICIAL NUBE MOLECULAR _________________________________________________
"""
DT = 1e5  # 27 años
TF = 1e10  # 27 000 años
OUTPUT_INTERVAL = DT * TF
NUM_STEPS = int(TF / DT)
sistemas= ["plummer_cluster"]
"""
#______________________________________________________________________________
# COND PARA UNA GALAXIA ESPIRAL________________________________________________
'''
TF = 500 #100000      
DT = 1     # 2
NUM_STEPS = int(TF / DT)   

OUTPUT_INTERVAL = 1
sistemas = ["soles_espiral"]
'''
#______________________________________________________________________________
#EL PROBLEMA DE LOS 3 CUERPOS NO RESTRINGIDO AL PLANO #########################

# COND INICIAL M3 NO COPLANARIA________________________________________________


TF = 90               # Tiempo final de simulación en segundos
DT = 0.001            # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT)  # Número total de pasos de integración

sistemas = ["random_3d_3_body"] 


#______________________________________________________________________________




###############################################################################

metodos = ["paso_rk5","paso_verlet_velocidades","paso_Euler",  "paso backward Euler"]
integradores = {"paso backward Euler": backward_euler,"paso_rk5": paso_rk5,"paso_Euler": paso_Euler,
 "paso_verlet_velocidades": paso_verlet_velocidades}


# Seleccionamos el método que queramos probar (por ejemplo, RK5)
for sistema in sistemas:
    # Si se quiere probar todos, usar: for metodo in metodos:
    for metodo in [metodos[0]]: 
        print(f"\nSimulando sistema: {sistema}  |  Método: {metodo}")

        # Inicializar el sistema
        system, labels, colors, legend = Cond_iniciales_definidas(sistema)

        # Array para aceleraciones
        a = np.zeros_like(system.x)

        # Arrays para guardar la trayectoria
        sol_x = np.zeros((NUM_STEPS+1, system.num_particulas, 3))
        sol_v = np.zeros((NUM_STEPS+1, system.num_particulas, 3))
        sol_t = np.zeros(NUM_STEPS+1)

        # Guardamos condiciones iniciales
        sol_x[0] = system.x.copy()
        sol_v[0] = system.v.copy()
        sol_t[0] = 0.0

        # Seleccionar integrador
        integrador = integradores[metodo]

        # Vamos a generarle un título a la señal 
        
        
        # Simulación
        t_inicio = time.time()
        print(NUM_STEPS)
        for step in range(1, NUM_STEPS+1):
            integrador(a, system, DT)
            sol_x[step] = system.x.copy()
            sol_v[step] = system.v.copy()
            sol_t[step] = step * DT
            

            if step % (NUM_STEPS//100) == 0:
                print(f"Simulación al {step*100//NUM_STEPS}% completada...")
                #titulo = f"{sistema} ({metodo}) ({step}%)"
                #plot_trayectoria(sol_x, labels, colors, legend, title=titulo)
                

        print(f"Cálculos terminados en {time.time() - t_inicio:.2f} s")

        # Vamos a generarle un título a cada simulación realizada :
        titulo = f"{sistema} ({metodo})"
        
        
        #REPRESENTACIONES  Y ANIMACIONES PARA CADA UNO DE LOS MÉTODOS: 
            
        '''
        Llegados a este punto vamos a utilizar todas las funciones que habíamos
        definido para estudiar nuestro sistema. Vamos a generar multitud de 
        figuras y animaciones, lo cual en algunos casos compromete el tiempo de 
        ejecución del programa. Es por ello, qe hemos decidido dejar algunas
        figuras comentadas, simplemente se quita el comentario segun las
        necesidades que tengamos en cada momento.
        '''
        #----------------------------------------------------------------------
        # 1. Plot  trayectoria 2D para ver lo que ha salido 
        plot_trayectoria(sol_x, labels, colors, legend, title=titulo)
        print("trayectoria en el plano XY hecha")
        
        #También podemos hacer aquí los GIFs : 
        #gif_name = f"{sistema}_{metodo}.gif"
        #make_gif_trajectory(sol_x,labels,colors,filename=gif_name,title=titulo,step=1000)
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # 2, Graficar condiciones iniciales para comprobar :  
        #plot_initial_conditions(system, labels, colors, legend, title=titulo)
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # 3. Análisis y estudio del error de la energía y del momento angular :
        #rel_error = compute_rel_energy_error(sol_x, sol_v, system)
        #plot_rel_energy_error(rel_error, sol_t, title=titulo)
        
        #rel_error_L = compute_rel_angular_momentum_error(sol_x, sol_v, system)
        #plot_rel_L_error(rel_error_L,sol_t,title=f"Error relativo en el momento angular — {titulo}")
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # 4. Estudio de las energías del sistema :
        print("Calculando energías para cada cuerpo...")
        T_ind, V_ind, E_ind = calcular_energias_individuales(sol_x, sol_v, system)
        T_sistema = np.sum(T_ind, axis=1)
        # Sumamos todas las potenciales (V) y DIVIDIMOS POR 2 (para no contar enlaces dobles)
        V_sistema = 0.5 * np.sum(V_ind, axis=1)
        # Energía Total (Esta sí debe ser constante)
        E_sistema = T_sistema + V_sistema
        
        #Llamamos a la función de plot modificada (pasándole esta nueva variable)
        plot_energias_con_total(E_ind, E_sistema, sol_t, labels, colors)
        #----------------------------------------------------------------------
        
        
        #----------------------------------------------------------------------
        # 5. GENERACIÓN DEL GIF 3D
        gif_name3D = f"{sistema}_{metodo}_3D.gif" # el nombre del archivo
        
        # Definimos cuántos fotogramas queremos en total
        total_frames_deseados = 200
        
        # Calculamos el 'step' (salto) para no procesar todos los puntos
        salto_ideal = max(1, int(NUM_STEPS / total_frames_deseados))
        
        print(f"Generando GIF 3D con un salto de {salto_ideal} pasos...")
        print(f"(Fotogramas totales: aprox {int(NUM_STEPS/salto_ideal)})")

        #Asignamos el resultado a una variable
        animacion_guardada = hacer_gif_trayectoria_3D(sol_x, labels, colors,
            title=titulo,filename=gif_name3D, step=salto_ideal) 
        #----------------------------------------------------------------------
        
        
        print("-" * 50)
        




###############################################################################
###############################################################################
###############################################################################


#10.ANEXOS Y CONFIGURACIONES PARA LA PRESENTACIÓN##############################

'''
Ya estudiado el sistema, para realizar la presentación se han preparado
diversas configuraciones reciclando elementos de código y graficando otras
magnitudes de interés. En estos anexos recogemos otras modificaciones del 
código que se han realizado para poner en la presentación diversos aspectos
del sistema como lo son el error o el intento de  las secciones de poincaré que 
las vamos a estudiar en el apartado de análsis del sistema.
'''


#ANÁLISIS CONJUNTO DEL ERROR DE E Y L__________________________________________
#Simplemente es para hacer las gráficas de error de E y L para los 4 métodos a
#la vez. En esta sección estudiaremos los errores qu luego ponemos en la 
#presentación 
"""
def comparar_errores_metodos(sistema,metodos,integradores,DT,NUM_STEPS,):
    # Diccionarios para almacenar los errores relativos y tiempos
    errores_E = {}
    errores_L = {}
    tiempos = {}

    for metodo in metodos:
        print(f"\nMétodo: {metodo}")

        # Inicializamos el sistema con las mismas condiciones iniciales
        system, labels, colors, legend = Cond_iniciales_definidas(sistema)
        a = np.zeros_like(system.x)

        # Generamos arrays completos (sin saltos)
        sol_x = np.zeros((NUM_STEPS + 1, system.num_particulas, 3))
        sol_v = np.zeros((NUM_STEPS + 1, system.num_particulas, 3))
        sol_t = np.zeros(NUM_STEPS + 1)

        # Guardamos las condiciones iniciales en el paso 0
        sol_x[0] = system.x.copy()
        sol_v[0] = system.v.copy()
        sol_t[0] = 0.0

        # Seleccionamos el integrador correspondiente al método actual
        integrador = integradores[metodo]

        t0 = time.time()
        for step in range(1, NUM_STEPS + 1):
            # Avanzamos el sistema un paso temporal DT
            integrador(a, system, DT)
            # Guardamos posiciones y velocidades tras el paso
            sol_x[step] = system.x.copy()
            sol_v[step] = system.v.copy()
            sol_t[step] = step * DT

            # Mostramos progreso cada 10% de la simulación
            if step % max(1, NUM_STEPS // 10) == 0:
                print(f"  {100 * step // NUM_STEPS}% completado")

        # Guardamos el tiempo total de ejecución del método
        tiempos[metodo] = time.time() - t0
        print(f"  Tiempo: {tiempos[metodo]:.2f} s")

        # Cálculo de errores
        errores_E[metodo] = compute_rel_energy_error(sol_x, sol_v, system)
        errores_L[metodo] = compute_rel_angular_momentum_error(sol_x, sol_v, system)

    # Dibujamos el error en energía para cada método
    plt.figure(figsize=(8, 5))
    for metodo in metodos:
        plt.plot(sol_t, errores_E[metodo], label=metodo)

    plt.yscale("log")
    plt.xlabel("Tiempo")
    plt.ylabel("Error relativo en energía")
    plt.title(f"Conservación de energía — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    # Dibujamos el error en momento angular para cada método
    plt.figure(figsize=(8, 5))
    for metodo in metodos:
        plt.plot(sol_t, errores_L[metodo], label=metodo)

    plt.yscale("log")
    plt.xlabel("Tiempo")
    plt.ylabel("Error relativo en momento angular")
    plt.title(f"Conservación del momento angular — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    return errores_E, errores_L, tiempos

TF = 200                # Tiempo final de simulación en segundos
DT = 0.008           # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["chaotic_3_body"]

metodos = ["paso backward Euler", "paso_verlet_velocidades", "paso_Euler", "paso_rk5"]
integradores = {"paso backward Euler": backward_euler,"paso_rk5": paso_rk5,"paso_Euler": paso_Euler,
 "paso_verlet_velocidades": paso_verlet_velocidades}

comparar_errores_metodos(sistema=sistemas[0],metodos=metodos,integradores=integradores,DT=DT,NUM_STEPS=NUM_STEPS)
"""
#______________________________________________________________________________

#####################CÁLCULO DEL ERROR Y TIEMPO SIMULACIÓN#####################
"""
def error_vs_dt_sistemas(sistema, metodos, integradores,DT_min, DT_max, num_dt, NUM_STEPS_ref,etiquetas_graficas=None, delta_dt=1e-4):

    print("COMPARACIÓN DE MÉTODOS NUMÉRICOS")
    print(f"Sistema: {sistema}")
    print(f"DT_min (referencia) = {DT_min}")
    print(f"DT_max = {DT_max}")
    print(f"num_dt = {num_dt}")
    print(f"NUM_STEPS_ref = {NUM_STEPS_ref}")
    print("Métodos:")
    for metodo in metodos:
        print(f"  - {metodo}")

    #1.) Solución de referencia (RK5)__________________________________________
    system_ref, labels, colors, legend = Cond_iniciales_definidas(sistema)   # Inicializamos el sistema de referencia
    a_ref = np.zeros_like(system_ref.x)                                      # Inicializamos un array de aceleraciones con ceros

    sol_x_ref = np.zeros((NUM_STEPS_ref + 1, system_ref.num_particulas, 3)) # Creamos array para guardar la trayectoria completa
    sol_x_ref[0] = system_ref.x.copy()                                        # Guardamos la posición inicial en la primera posición del array


    print(f"Generando solución de referencia con RK5 ({NUM_STEPS_ref} pasos)...")
    for step in range(1, NUM_STEPS_ref + 1):                                 # Iteramos sobre todos los pasos de la referencia
        integradores["paso_rk5"](a_ref, system_ref, DT_min)                   # Realizamos un paso RK5
        sol_x_ref[step] = system_ref.x.copy()                                 # Guardamos la posición actual
        if step % max(1, NUM_STEPS_ref // 10) == 0:
            print(f"  Referencia RK5: {100 * step / NUM_STEPS_ref:.0f}%")

    #2.) dts a estudiar________________________________________________________
    dts = np.linspace(DT_min + delta_dt, DT_max, num_dt)

    #3.) Inicialización de errores y tiempos___________________________________
    errores_max = [[] for metodo in metodos]                                        # Lista de errores máximos globales por método
    errores_med = [[] for metodo in metodos]                                        # Lista de errores medios globales por método
    tiempos = [[] for metodo in metodos]                                            # Lista de tiempos de simulación por método


    errores_max_body = [[[] for _ in range(system_ref.num_particulas)]for metodo in metodos]       # Lista de errores máximos por cuerpo
    errores_med_body = [[[] for metodo in range(system_ref.num_particulas)]for metodo in metodos]  # Lista de errores medios por cuerpo

    # Bucle principal
    for indice_dt, dt in enumerate(dts):
        NUM_STEPS = int(NUM_STEPS_ref * DT_min / dt)                            # Calculamos el número de pasos necesarios para este dt

        print(f"dt {indice_dt + 1}/{num_dt} = {dt:.3e}")
        print(f"NUM_STEPS = {NUM_STEPS}")

        #4.) Bucle sobre cada método__________________________________________
        for indice_metodo, metodo in enumerate(metodos):
            NUM_STEPS = max(1, int(NUM_STEPS_ref * DT_min / dt))                # Aseguramos al menos un paso
            print(f"\nMétodo: {metodo}")

            system_test, labels, colors, legend = Cond_iniciales_definidas(sistema)  # Inicializamos sistema para este método
            a_test = np.zeros_like(system_test.x)                                   # Inicializamos aceleraciones en cero

            sol_x_test = np.zeros((NUM_STEPS + 1, system_test.num_particulas, 3))  # Creamos array para guardar trayectoria
            sol_x_test[0] = system_test.x.copy()                                     # Guardamos posición inicial

            t0 = time.time()                                                        # Registramos tiempo de inicio

            for step in range(1, NUM_STEPS + 1):                                    # Iteramos sobre los pasos
                integradores[metodo](a_test, system_test, dt)                        # Realizamos un paso del método actual
                sol_x_test[step] = system_test.x.copy()                              # Guardamos posición actual

                if step % max(1, NUM_STEPS // 10) == 0:
                    print(f"  Progreso integración: {100 * step / NUM_STEPS:.0f}%")

            tiempos[indice_metodo].append(time.time() - t0)

            #5.) Interpolación de la solución de referencia____________________
            factor = NUM_STEPS_ref / NUM_STEPS
            indices_ref = (np.arange(NUM_STEPS + 1) * factor).astype(int)
            sol_ref_interp = sol_x_ref[indices_ref]

            #6.) Errores_______________________________________________________           
            err = np.linalg.norm(sol_x_test - sol_ref_interp, axis=2)                 # Calculamos error por paso y cuerpo

            errores_max[indice_metodo].append(np.max(err))                             # Guardamos error máximo global
            errores_med[indice_metodo].append(np.mean(err))                             # Guardamos error medio global

            for cuerpo in range(system_test.num_particulas):                          # Iteramos sobre cada cuerpo
                errores_max_body[indice_metodo][cuerpo].append(
                    np.max(err[:, cuerpo])                                             # Guardamos error máximo de cada cuerpo
                )
                errores_med_body[indice_metodo][cuerpo].append(
                    np.mean(err[:, cuerpo])                                            # Guardamos error medio de cada cuerpo
                )


    #7.) Graficamos de error global____________________________________________
    plt.figure(figsize=(8, 5))
    for i, metodo in enumerate(metodos):
        label = etiquetas_graficas[metodo] if etiquetas_graficas else metodo
        #ejes logaritmicos
        plt.loglog(dts, errores_max[i], '-', label=f"{label} error máx")
        plt.loglog(dts, errores_med[i], '--', label=f"{label} error medio")
        #ejes no logaritmicos
        #plt.plot(dts, errores_max[i], '-', label=f"{label} error máx")
        #plt.plot(dts, errores_med[i], '--', label=f"{label} error medio")
    plt.xlabel("dt")
    plt.ylabel("Error global")
    plt.title(f"Error global vs dt — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    for i, metodo in enumerate(metodos):
        label = etiquetas_graficas[metodo] if etiquetas_graficas else metodo
        #ejes logaritmicos
        #plt.loglog(dts, errores_max[i], '-', label=f"{label} error máx")
        plt.loglog(dts, errores_med[i], '--', label=f"{label}")
        #ejes no logaritmicos
        #plt.plot(dts, errores_max[i], '-', label=f"{label} error máx")
        #plt.plot(dts, errores_med[i], '--', label=f"{label} error medio")
    plt.xlabel("dt")
    plt.ylabel("Error global")
    plt.title(f"Error medio global vs dt — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 5))
    for i, metodo in enumerate(metodos):
        label = etiquetas_graficas[metodo] if etiquetas_graficas else metodo
        #ejes logaritmicos
        plt.loglog(dts, errores_max[i], '-', label=f"{label}")
        #plt.loglog(dts, errores_med[i], '--', label=f"{label} error medio")
        #ejes no logaritmicos
        #plt.plot(dts, errores_max[i], '-', label=f"{label} error máx")
        #plt.plot(dts, errores_med[i], '--', label=f"{label} error medio")
    plt.xlabel("dt")
    plt.ylabel("Error global")
    plt.title(f"Error maximo global vs dt — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    #8.) Graficamos de error por cuerpo________________________________________
    for cuerpo in range(system_ref.num_particulas):
        plt.figure(figsize=(8, 5))
        for i, metodo in enumerate(metodos):
            label = etiquetas_graficas[metodo] if etiquetas_graficas else metodo
            plt.loglog(dts, errores_max_body[i][cuerpo], '-',label=f"{label} error máx")
            plt.loglog(dts, errores_med_body[i][cuerpo], '--',label=f"{label} error medio")
            #ejes no logaritmicos
            #plt.plot(dts, errores_max_body[i][cuerpo], '-', label=f"{label} error máx")
            #plt.plot(dts, errores_med_body[i][cuerpo], '--', label=f"{label} error medio")

        plt.xlabel("dt")
        plt.ylabel("Error")
        plt.title(f"Error vs dt — {sistema} — {labels[cuerpo]}")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

    #9.) Graficamos tiempos de simulación______________________________________
    plt.figure(figsize=(8, 5))
    for i, metodo in enumerate(metodos):
        label = etiquetas_graficas[metodo] if etiquetas_graficas else metodo
        plt.loglog(dts, tiempos[i], '-', label=label)
        #plt.plot(dts, tiempos[i], '-', label=label)

    plt.xlabel("dt")
    plt.ylabel("Tiempo de simulación (s)")
    plt.title(f"Tiempos de simulación vs dt — {sistema}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

#TF = 3                 # Tiempo final de simulación en segundos
#DT = 0.000005              # Paso de tiempo en segundos
#OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
#NUM_STEPS = int(TF / DT) # Número total de pasos de integración

#sistemas = ["skinny_pineapple_3_body"]
TF = 200                # Tiempo final de simulación en segundos
DT = 0.0001           # Paso de tiempo en segundos
OUTPUT_INTERVAL = 1      # Intervalo de salida de resultados en segundos
NUM_STEPS = int(TF / DT) # Número total de pasos de integración

sistemas = ["chaotic_3_body"]

metodos = ["paso_verlet_velocidades", "paso_Euler", "paso_rk5", "paso backward Euler"]
integradores = {
    "paso_rk5": paso_rk5,
    "paso_Euler": paso_Euler,
    "paso_verlet_velocidades": paso_verlet_velocidades,
    "paso backward Euler": backward_euler
}
etiquetas_graficas = {
    "paso_verlet_velocidades": "Verlet",
    "paso_Euler": "Euler",
    "paso_rk5": "rk5",
    "paso backward Euler": "Backward Euler"
}
DT_min=0.0001

error_vs_dt_sistemas(
    sistema=sistemas[0],
    metodos=metodos,
    integradores=integradores,
    DT_min=DT_min,        # dt de la solución perfecta
    DT_max=0.1,
    num_dt=100,
    NUM_STEPS_ref=int(TF / DT_min),
    etiquetas_graficas=etiquetas_graficas,
    delta_dt=0.001        # para que los dt evaluados estén un poco por encima de la referencia
)

"""
#______________________________________________________________________________





# FUNCION QUE CALCULA LA SECCIÓN DE POINCARÉ que se utiliza en la presentacion
# a modo de ejemplo. 
'''
# 1. FUNCIÓN DE INTEGRACIÓN DE POINCARÉ (OPTIMIZADA) --------------------------
def seccion_poincare(system: "System", dt: float, tf: float, 
                     cuerpo_perturbado: int = None, delta: float = 0.0, 
                     report_interval: int = 50000):
    """
    Calcula la Sección de Poincaré para el plano Y=0 (cruce ascendente).
    Utiliza interpolación lineal para mayor precisión en el punto de corte.
    """
    num_pasos = int(tf / dt)
    a = np.zeros_like(system.x)
    puntospoincare = []

    # --- APLICAR PERTURBACIÓN (Si se solicita) ---
    if cuerpo_perturbado is not None:
        system.v[cuerpo_perturbado, 0] += delta
        print(f"--> Perturbación aplicada: {delta} a vx del cuerpo {cuerpo_perturbado}")

    # --- INICIALIZACIÓN DE MEMORIA ---
    prev_y = system.x[:, 1].copy()   # Y anterior
    prev_x = system.x[:, 0].copy()   # X anterior
    prev_vx = system.v[:, 0].copy()  # Vx anterior

    print(f"Iniciando integración Poincaré (TF={tf}, DT={dt})...")

    for paso in range(num_pasos):
        # 1. Avanzamos un paso con RK5
        paso_rk5(a, system, dt)

        # 2. Detectamos cruces por el plano Y=0
        for i in range(system.num_particulas):
            y_actual = system.x[i, 1]
            y_anterior = prev_y[i]

            # CONDICIÓN DE POINCARÉ:
            # Cruce de negativo a positivo (y_ant < 0 y y_act >= 0)
            if y_anterior < 0 and y_actual >= 0:
                
                # INTERPOLACIÓN LINEAL
                # Encontramos el momento exacto donde y=0 entre los dos pasos
                denom = (y_actual - y_anterior)
                fraction = -y_anterior / denom if denom != 0 else 0
                
                # Calculamos X y Vx exactos en ese instante
                x_cross = prev_x[i] + fraction * (system.x[i, 0] - prev_x[i])
                vx_cross = prev_vx[i] + fraction * (system.v[i, 0] - prev_vx[i])

                # Guardamos [x, vx]
                puntospoincare.append([x_cross, vx_cross])

        # 3. Actualizamos memoria
        prev_y = system.x[:, 1].copy()
        prev_x = system.x[:, 0].copy()
        prev_vx = system.v[:, 0].copy()

        # Reporte de progreso
        if paso % report_interval == 0 and paso > 0:
             pct = paso / num_pasos * 100
             print(f"   Progreso: {pct:.1f}% | Puntos encontrados: {len(puntospoincare)}")

    print(f"Terminado. Total puntos Poincaré: {len(puntospoincare)}")
    return np.array(puntospoincare)

# -----------------------------------------------------------------------------
# 2. BLOQUE PRINCIPAL DE EJECUCIÓN (SOLO SISTEMA BASE)
# -----------------------------------------------------------------------------

# A) Parámetros de simulación
TF = 1000           # Tiempo final
DT = 1e-4          # Paso de tiempo
SISTEMA = "chaotic_3_body" # O cambia a "lagrange_3_body" según necesites

print("==========================================================")
print(f"CALCULANDO SECCIÓN DE POINCARÉ (BASE): {SISTEMA}")
print("==========================================================")

# B) Cálculo del Sistema Original
sys_orig, _, _, _ = Cond_iniciales_definidas(SISTEMA)
pts_orig = seccion_poincare(sys_orig, DT, TF)

# C) Representación Gráfica (Solo una figura)
print("\nGenerando gráfico...")

plt.figure(figsize=(10, 8)) # Una sola figura grande

if len(pts_orig) > 0:
    # Graficamos los puntos (x vs vx)
    plt.scatter(pts_orig[:, 0], pts_orig[:, 1], s=15, c='blue', alpha=0.6)
else:
    plt.text(0, 0, "Sin cruces detectados (Aumenta TF)", ha='center')

plt.title(f"Sección de Poincaré - {SISTEMA} (TF={TF})")
plt.xlabel("$x$ (Posición)")
plt.ylabel("$v_x$ (Velocidad)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''
###############################################################################
###############################################################################
###############################################################################