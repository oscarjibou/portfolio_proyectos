*-----------------------------------------------------------
* Title      :
* Written by :
* Date       :
* Description:
*-----------------------------------------------------------   
    ORG    $1000

START: 

    MOVE.B #5,M1
    MOVE.B #7,M2  
    SUB.L M2,M1

* Put program code here


* Put variables and constants here

    END   START
    
    ORG $3000
DOS:
    MOVE.B $2000,M1
    MOVE.B $2001,M2
    MOVE.B M1,R1
    ADD.B M2,M1
    MOVE.B R1,M1
    SUB.B M2,M1
    
    END DOS
    
*~Font name~Courier New~
*~Font size~10~
*~Tab type~1~
*~Tab size~4~
