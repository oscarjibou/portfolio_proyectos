*-----------------------------------------------------------
* Title      : Practica 3
* Written by : Carlos E. Dominguez Martinez
* Date       : 2/3/23
* Instrucciones: Los ejercicios empiezan en la posicion de 
* memoria 2000, y cambian cada 1000 posiciones. De forma
* que si quiere ir al ejercicio x, deberá comencar el PC en:
* (nº ejercicio + 1)000. 
*-----------------------------------------------------------
    ORG    $1000
    *Datos1
MAYOR   CMP.L D0,D1
        BPL FIN1
        MOVE.L D0,D1
        RTS   
        
     *Datos2
        
MAY     CMP.L D0,D1
        BPL TEXTO
        MOVE.L D0,D1
        RTS
        
MENSAJE1 DC.B 'Primer numero>',0
MENSAJE2 DC.B 'Segundo numero>',0
MENSAJE3 DC.B 'El mayor de los numeros es>',0 


    *Datos3y4
LED     EQU $E00010
PUL     EQU $E00014
V1      DC.B 128
V2      DC.B 1  
        
   * EJ1 
               
    ORG $2000
START:  
        MOVE.L #20,D0
        MOVE.L #5,D1
        JSR MAYOR
FIN1
    JMP $3000
    
    * EJ2 
    
    ORG $3000
        MOVE.L #14,D0
        MOVE.L #MENSAJE1,A1
        TRAP #15

        MOVE.L #4,D0
        TRAP #15
        MOVE.L D1,D2
        
        MOVE.L #14,D0
        MOVE.L #MENSAJE2,A1
        TRAP #15   
        
        MOVE.L #4,D0
        TRAP #15
        
        MOVE.L D2,D0   
        CLR.L D2
        JSR MAY
        
TEXTO   MOVE.L #14,D0
        MOVE.L #MENSAJE3,A1
        TRAP #15
        
        MOVE.L #3,D0
        TRAP #15
FIN2                
        JMP $4000

    * EJ3 
        
        ORG   $4000
        MOVE.B V1,LED
        MOVE.B V1,D2      
       
D       LSR.B #1, D2        ; Desplazar LEDs a la derecha 
        MOVE.B D2, LED
        JSR DELAY           ; Escribir el nuevo valor de los LEDs
        CMP.B  V2,D2 
        BEQ IZ
        JMP D
        
IZ      LSL.B #1, D2        ; Desplazar LEDs a la derecha
        MOVE.B D2, LED
        JSR DELAY           ; Escribir el nuevo valor de los LEDs
        CMP.B  V1,D2 
        BEQ D
        JMP IZ
DELAY:
        MOVE.L #23, D0 
        MOVE.L #5, D1
        TRAP #15
        RTS
        
        *EJ4
        
        ORG $5000
        MOVE.B #0,LED
        MOVE.B #0,PUL
IN      MOVE.B #0,D1
        
COM     MOVE.B PUL,D2
        CMP.B #$FF,LED
        BEQ ENC 
        BTST.L D1,D2
        BNE EN
        ADD.B #1,D1
        JSR MAX
        
        JMP COM
        
EN      BSET.L D1,D2
        BCLR.L D1,D1
        MOVE.B D2,LED
        JMP COM
 
MAX     CMP.B #8,D1
        BEQ IN
        RTS
        
ENC    
        MOVE.B #0,LED 
        
       END    START        ; last line of source






*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
