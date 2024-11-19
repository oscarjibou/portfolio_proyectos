*-----------------------------------------------------------
* Title      : 2ª Sesión de la Práctica
* Written by : Carlos E. Dominguez Martinez
* Date       :
* Description:
*-----------------------------------------------------------
    ORG    $1000
    
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
    
    ORG $2000   
    
START: 
   
    MOVE.L #0, D1
    MOVE.L #0, D2
    MOVE.L #1, D3
    JSR CNT1
    
CNT1
    MOVE.L #100000,D5
    JSR REPITE
    MOVE.L D2,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E00000
    ADD.L D3, D2
    CMP #10, D2
    BCS CNT1
    MOVE.L #8, D2 
    BCC CNT2
    
CNT2
    MOVE.L #100000,D5
    JSR REPITE
    MOVE.L D2,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E00000
    SUB.L D3, D2
    CMP #0, D2
    BEQ CNT1
    BCC CNT2
    
REPITE 
    NOP
    SUB.L #1, D5
    CMP.L #0,D5
    BNE REPITE
    RTS

DELAY:
        MOVE.L #23, D0 
        MOVE.L #5, D1
        TRAP #15
        RTS
        

  

* Put variables and constants here

    END    START        ; last line of source





*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
