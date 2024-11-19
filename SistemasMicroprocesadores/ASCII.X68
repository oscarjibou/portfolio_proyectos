*-----------------------------------------------------------
* Title      : Conversor de cadenas en ASCII
* Written by : Carlos E. Dominguez Martinez
* Date       : 24/2/2023
* Description: He a�adido un par de escenarios extra, como 
* los numeros negativos o que no sean numeros.
*-----------------------------------------------------------
    
      ORG    $1000        
    *Datos del programa
SP    DC.B   '-34a',0    * Cadena ASCII a convertir
EP    DC.B   'Error',0   * Mensaje de error


START  
       MOVE.W #SP,A0    * Puntero a la cadena ASCII
       CLR.L  D0        * Borrar D0, ya que se almacenar� ah� el resultado
       MOVE.B (A0)+,D1  * Cargar el primer car�cter de la cadena
       CMP.B  #'-',D1   * �Es un signo menos?
       BNE    NORM      * Si no, saltar al an�lisis normal
       MOVE.B (A0)+,D1  * Cargar el siguiente car�cter
       NEG.L  D0        * Negar el resultado final
NORM   CMP.B  #0,D1     * �Fin de la cadena?
       BEQ    FIN       * Si es as�, salir del bucle
       CMP.B  #'0',D1   * �Es un d�gito?
       BLT    ER        * Si no, mostrar un error
       CMP.B  #'9',D1
       BGT    ER
       MULU   #10,D0    * Multiplicar el resultado actual por 10
       SUB.B  #'0',D1   * Convertir el d�gito ASCII en un n�mero decimal
       ADD.L  D1,D0     * Sumar el d�gito al resultado actual
       MOVE.B (A0)+,D1  * Cargar el siguiente car�cter
       BRA    NORM      * Volver al an�lisis
ER     MOVE.L #EP,D1    * Mostrar mensaje de error
       BRA    FINE      * Salir del programa                
FINE   MOVE.L #0,D0     * Colocar cero en D0 para indicar error
FIN  
    END    START        

*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
