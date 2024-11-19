*-----------------------------------------------------------
* Title      :
* Written by :
* Date       :
* Description:
*-----------------------------------------------------------
    ORG    $1000
START:                  ; first instruction of program
    JMP RESET

* Put program code here
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

    SIMHALT             ; halt simulator
   
SEGUNDO1
*    MOVE.L #10000,D0
*    JSR REPITE
    JSR PANTALLA1
    ADD.L #1, D2
    CMP #10, D2
    BCS SEGUNDO1
    BCC SEGUNDO2
    
SEGUNDO2
*    MOVE.L #10000,D0
*    JSR REPITE
    MOVE.L #0, D2
    JSR PANTALLA1
    ADD.L #1, D3
    JSR PANTALLA2
    CMP #6, D3
    BCS SEGUNDO1
    BCC MINUTO3

MINUTO3
*    MOVE.L #10000,D0
*    JSR REPITE
    MOVE.L #0,D2
    JSR PANTALLA1
    MOVE.L #0, D3
    JSR PANTALLA2
    ADD.L #1, D4
    JSR PANTALLA3
    CMP #10, D4
    BCS SEGUNDO1
    BCC MINUTO4

MINUTO4
*    MOVE.L #10000,D0
*    JSR REPITE
    
    MOVE.L #0, D2
    JSR PANTALLA1
    MOVE.L #0, D3
    JSR PANTALLA2
    MOVE.L #0, D4
    JSR PANTALLA3
    ADD.L #1, D5
    JSR PANTALLA4
    CMP #6, D5
    BCS SEGUNDO1
    CMP #2, D7
    BCS HORA5
    BCC FINDIA

    
HORA5 
*    MOVE.L #10000,D0
*    JSR REPITE
    MOVE.L #0,D2
    JSR PANTALLA1
    MOVE.L #0, D3
    JSR PANTALLA2
    MOVE.L #0, D4
    JSR PANTALLA3
    MOVE.L #0, D5
    JSR PANTALLA4
    ADD.L #1, D6
    JSR PANTALLA5
    CMP #10, D6
    BCS SEGUNDO1
    BCC HORA6
        
HORA6
*    MOVE.L #10000,D0
*    JSR REPITE
    MOVE.L #0,D2
    JSR PANTALLA1
    MOVE.L #0, D3
    JSR PANTALLA2
    MOVE.L #0, D4
    JSR PANTALLA3
    MOVE.L #0, D5
    JSR PANTALLA4
    MOVE.L #0, D6
    JSR PANTALLA5
    ADD.L #1, D7
    JSR PANTALLA6
    JMP SEGUNDO1

FINDIA
*    MOVE.L #10000,D0
*    JSR REPITE
    MOVE.L #0,D2
    JSR PANTALLA1
    MOVE.L #0, D3
    JSR PANTALLA2
    MOVE.L #0, D4
    JSR PANTALLA3
    MOVE.L #0, D5
    JSR PANTALLA4    
    ADD.L #1, D6
    JSR PANTALLA5
    CMP #4, D6
    BCS SEGUNDO1
    JMP RESET


RESET
    MOVE.L #0, D1 
    MOVE.L #0, D2 ;pos1
    MOVE.L #0, D3 ;pos2
    MOVE.L #0, D4 ;pos3
    MOVE.L #0, D5 ;pos4
    MOVE.L #0, D6 ;pos5
    MOVE.L #0, D7 ;pos6
    JSR PANTALLA1
    JSR PANTALLA2
    JSR PANTALLA3
    JSR PANTALLA4
    JSR PANTALLA5
    JSR PANTALLA6
    JMP SEGUNDO1

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
PANTALLA3
    MOVE.L D4,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E0000A
    RTS
PANTALLA4
    MOVE.L D5,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E00008
    RTS
PANTALLA5
    MOVE.L D6,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E00006
    RTS
PANTALLA6
    MOVE.L D7,D1
    MOVE.L #TABLA,A1
    ADD.L D1,A1
    MOVE.B (A1),$E00004
    RTS
    
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
