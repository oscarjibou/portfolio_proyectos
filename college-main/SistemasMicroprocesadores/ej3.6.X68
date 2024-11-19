*-----------------------------------------------------------
* Title      :
* Written by :
* Date       :
* Description:
*-----------------------------------------------------------
    ORG    $1000
START:                  ; first instruction of program
* Ejemplo para sacar un 6
    JSR RESET
    
*DATOS
TABLA 
    DC.B $3F DIGITO 0
    DC.B $06 DIGITO 1
    DC.B $5B DIGITO 2
    DC.B $4F DIGITO 3
    DC.B $66 DIGITO 4
    DC.B $6D DIGITO 5
    DC.B $7D DIGITO 6
    DC.B $07 DIGITO 7
    DC.B $7F DIGITO 8
    DC.B $6F DIGITO 9
    DC.B $00 DIGITO OFF
    DC.B $80 PUNTO
* Put program code here


    SIMHALT             ; halt simulator

UNIDAD
    MOVE.L #10000,D0
    JSR REPITE
    JSR PANTALLA1
    ADD.L #1, D2
    CMP #10, D2
    BCS UNIDAD
    BCC DECENA
    
DECENA
    MOVE.L #10000,D0
    JSR REPITE
    MOVE.L #0, D2
    JSR PANTALLA1
    ADD.L #1, D3
    JSR PANTALLA2
    CMP #6, D3
    BCS UNIDAD
    BCC RESET

PANTALLA1 
    MOVE.L D2,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E0000E
    RTS
PANTALLA2
    MOVE.L D3,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E0000C
    RTS
    
RESET
    MOVE.L #0, D1
    MOVE.L #0, D2
    MOVE.L #0, D3
    JSR PANTALLA1
    JSR PANTALLA2
    JSR UNIDAD
    
REPITE 
    NOP
    SUB.L #1, D0
    CMP.L #0,D0
    BNE REPITE
    RTS


  

* Put variables and constants here

    END    START        ; last line of source





*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
