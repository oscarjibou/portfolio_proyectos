*-----------------------------------------------------------
* Title      : Practica 2
* Written by : Carlos E. Dominguez
* Date       : 24/2/2023
* Description: Los ejercicios están todos concatenados de 
* manera que si ejecuta el programa en continuo se iran 
* realizando todos los ejercicios propuestos.
*-----------------------------------------------------------

        *DATOS
    ORG $1000
    
DATOS   DC.B 9,6,7 *Datos ej1
FIND
DATOS2  DC.W 9,6,7 *Datos ej2
FIND2 
DATO1   DC.L 5     *Datos ej7
DATO2   DC.L 4     *Datos ej7
DATO3   DC.L 55    *Datos ej8
DATO4   DC.L 45    *Datos ej8
DATOS3  DC.B 9,6,7,1,4 Datos ej9
FIND3


    *OPERACIONES
OP      DC.L 1    *Condicional ej7
SUMA    ADD.L D0,D1
        JMP NEXT
RESTA   SUB.L D0,D1
        JMP NEXT  

RESTA1  SUB.L D1,D0 *Resta negativa
        JMP NEXT1
RESTA2  SUB.L D0,D1 *Resta positiva
        JMP NEXT1
        
* EJ1
    ORG $2000
START:                  
    MOVE.L #0,D0
    MOVE.L #0,D1
    MOVE.L #DATOS,A0
    MOVE.B (A0)+,D1
    ADD.L D1,D0
    MOVE.B (A0)+,D1
    ADD.L D1,D0

    JMP $3000
   
* EJ2 
    ORG $3000
    MOVE.L #0,D0
    MOVE.L #0,D1
    MOVE.L #DATOS2,A0
    MOVE.W (A0)+,D1
    ADD.L D1,D0
    MOVE.W (A0)+,D1
    ADD.L D1,D0
    
    JMP $4000
    
* EJ3 DECREMENTADOS
    ORG $4000
    MOVE.L #0,D0
    MOVE.L #0,D1
    MOVE.L #FIND,A0
    MOVE.B -(A0),D1
    ADD.L D1,D0
    MOVE.B -(A0),D1
    ADD.L D1,D0

    JMP $5000
   
* EJ4 DECREMENTOS
    ORG $5000
    MOVE.L #0,D0
    MOVE.L #0,D1
    MOVE.L #FIND2,A0
    MOVE.W -(A0),D1
    ADD.L D1,D0
    MOVE.W -(A0),D1
    ADD.L D1,D0

    JMP $7000
    
*EJ5 BUCLE INCONDICIONAL
    ORG $6000
        MOVE.L #0,D0
        MOVE.L #0,D1
        MOVE.L #DATOS,A0
BUCLE   MOVE.B (A0)+,D1
        ADD.L D1,D0
        MOVE.B (A0)+,D1
        ADD.L D1,D0
        JMP BUCLE
        
* EJ6 BUCLE CONDICIONAL
    ORG $7000
        MOVE.W #0,D0 *Contador
        MOVE.W #0,D1
BUCLE1
        ADD.W #1,D0
        ADD.W D0,D1
        CMP.W #100,D0        
        BNE BUCLE1
        
        JMP $8000      

* EJ7 SUMAS Y RESTAS CONDICIONALES       
    ORG $8000
        MOVE.L #0,D0
        MOVE.L #0,D1
        MOVE.L #DATO1,A0
        MOVE.L #DATO2,A1
        MOVE.L #OP,A2
        MOVE.L (A0),D0
        MOVE.L (A1),D1
        MOVE.L (A2),D2
        CMP.L #0,D2
        BEQ SUMA
        BNE RESTA
        
NEXT    JMP $9000

*EJ8 RESTADOR POSITIVO
    ORG $9000
        MOVE.L #0,D0
        MOVE.L #0,D1
        MOVE.L #DATO3,A0
        MOVE.L #DATO4,A1
        MOVE.L (A0),D0
        MOVE.L (A1),D1
        CMP.L D0,D1
        BPL.L RESTA2
        BMI RESTA1
        
NEXT1   JMP $10000

*EJ9 SUMADOR DE 3 VALORES
    ORG $10000
    MOVE.L #0,D0
    MOVE.L #0,D1
    MOVE.L #DATOS3,A0
S   MOVE.B (A0)+,D1
    ADD.L D1,D0
    CMP.L #FIND3,A0      
    BNE S               *Sigue sumando
    BEQ NEXT2
    
NEXT2   
    
    END    START        * Fin del programa



*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
