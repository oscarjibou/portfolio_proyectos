	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 * Title : Reloj Casio
 * Fecha : 29 / 05 / 23
 * Autores : Oscar Jimenez Bou y Carlos E. Dominguez Martinez
 * Descripcion : Reloj Casio con alarma y cronometro, ha de ser ejecutado en el emulador easy68k con los .wav en la misma carpeta.
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * INTERRUPCION
	ORG 25 * 4
	DC.L INT_CLK
	
	ORG $1000
	
	 * DATOS
	 * Datos mostrados por pantalla
	PS1 DS.W 1
	PS2 DS.W 1
	PM1 DS.W 1
	PM2 DS.W 1
	PH1 DS.W 1
	PH2 DS.W 1
	
	 * Variables Contador Generico
	CM1 DS.W 1
	CM2 DS.W 1
	CH1 DS.W 1
	CH2 DS.W 1
	
	 * Variables Alarma
	MN1ALMG DS.W 1
	MN2ALMG DS.W 1
	HR1ALMG DS.W 1
	HR2ALMG DS.W 1
	
	 * Variables Interrupcion
	SG1 DS.W 1
	SG2 DS.W 1
	MN1 DS.W 1
	MN2 DS.W 1
	HR1 DS.W 1
	HR2 DS.W 1
	
	
	 * Variables Contador Cronometro
	CS1CRON DS.W 1
	CS2CRON DS.W 1
	CM1CRON DS.W 1
	CM2CRON DS.W 1
	CH1CRON DS.W 1
	CH2CRON DS.W 1
	
	 * Variables Cronometro
	SG1CRON DS.W 1
	SG2CRON DS.W 1
	MN1CRON DS.W 1
	MN2CRON DS.W 1
	HR1CRON DS.W 1
	HR2CRON DS.W 1
	
	 * Variables Cronometro congeladas
	SG1CRONFREEZE DS.W 1
	SG2CRONFREEZE DS.W 1
	MN1CRONFREEZE DS.W 1
	MN2CRONFREEZE DS.W 1
	HR1CRONFREEZE DS.W 1
	HR2CRONFREEZE DS.W 1
	
	 * Variables SET
	MN1ST DS.W 1
	MN2ST DS.W 1
	HR1ST DS.W 1
	HR2ST DS.W 1
	
	 * CODIFICACION DIGITOS
	TABLA * %HGFEDCBA
	DC.B %00111111 DIGITO 0
	DC.B %00000110 DIGITO 1
	DC.B %01011011 DIGITO 2
	DC.B %01001111 DIGITO 3
	DC.B %01100110 DIGITO 4
	DC.B %01101101 DIGITO 5
	DC.B %01111101 DIGITO 6
	DC.B %00000111 DIGITO 7
	DC.B %01111111 DIGITO 8
	DC.B %01101111 DIGITO 9
	DC.B %00000000 APAGADO
	TABLA_P * %HGFEDCBA
	DC.B %10111111 DIGITO 0
	DC.B %10000110 DIGITO 1
	DC.B %11011011 DIGITO 2
	DC.B %11001111 DIGITO 3
	DC.B %11100110 DIGITO 4
	DC.B %11101101 DIGITO 5
	DC.B %11111101 DIGITO 6
	DC.B %10000111 DIGITO 7
	DC.B %11111111 DIGITO 8
	DC.B %11101111 DIGITO 9
	DC.B %00000000 APAGADO
	 * 7 - SEGMENTOS
	
	MODO1 EQU $E0000E
	MODO2 EQU $E0000C
	S1 EQU $E0000A
	S2 EQU $E00008
	M1 EQU $E00006
	M2 EQU $E00004
	H1 EQU $E00002
	H2 EQU $E00000
	
	 * LUCES LEDS
	LEDS EQU $E00010
	
	 * BOTONES
	BT EQU $E00014
	
	FLAGCRON DS.B 1
	FLAGALM DS.B 1
	FLAGALMSOUND DS.B 1
	 * SONIDOS
	
	ALMSOUND DC.B 'Alarma.wav', $00 * Sonido de la alarma
	BEEP DC.B 'Beep.wav', $00 * Sonido botones
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	
	ORG $2000
START: * Inizializamos todas las variables a 0 menos los FLAGS
	 * Variables Pantalla
	MOVE.B #0, PS1
	MOVE.B #0, PS2
	MOVE.B #0, PM1
	MOVE.B #0, PM2
	MOVE.B #0, PH1
	MOVE.B #0, PH2
	
	 * Variables Contador Generico
	MOVE.B #0, CM1
	MOVE.B #0, CM2
	MOVE.B #0, CH1
	MOVE.B #0, CH2
	
	 * Variables Alarma
	MOVE.B #0, MN1ALMG
	MOVE.B #0, MN2ALMG
	MOVE.B #0, HR1ALMG
	MOVE.B #0, HR2ALMG
	
	 * Variables Interrupcion
	MOVE.B #0, SG1
	MOVE.B #0, SG2
	MOVE.B #0, MN1
	MOVE.B #0, MN2
	MOVE.B #0, HR1
	MOVE.B #0, HR2
	
	 * Variables Contador Cronometro
	MOVE.B #0, CS1CRON
	MOVE.B #0, CS2CRON
	MOVE.B #0, CM1CRON
	MOVE.B #0, CM2CRON
	MOVE.B #0, CH1CRON
	MOVE.B #0, CH2CRON
	
	 * Variables Cronometro
	MOVE.B #0, SG1CRON
	MOVE.B #0, SG2CRON
	MOVE.B #0, MN1CRON
	MOVE.B #0, MN2CRON
	MOVE.B #0, HR1CRON
	MOVE.B #0, HR2CRON
	
	 * Variables SET
	MOVE.B #0, MN1ST
	MOVE.B #0, MN2ST
	MOVE.B #0, HR1ST
	MOVE.B #0, HR2ST
	
	JSR RST_PANT
	MOVE.B #0, LEDS * Luces apagadas
	
	MOVE.B #0, FLAGCRON
	MOVE.B #1, FLAGALM * #1 = MODO ALARMA ON
	MOVE.B #0, FLAGALMSOUND * #1 = SONIDO Y LUCES ON
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * RELOJ
	CLK
	 * Mostramos modo CLK
MOVE.B #%00111000, MODO1 * Letra de la Derecha: C
MOVE.B #%00111001, MODO2 * Letra de la Izquierda: L
	
	 * Pasamos las variables de INT_CLK a las de Pantalla
	MOVE.B (SG1), PS1
	MOVE.B (SG2), PS2
	MOVE.B (MN1), PM1
	MOVE.B (MN2), PM2
	MOVE.B (HR1), PH1
	MOVE.B (HR2), PH2
	 * Mostramos por pantalla esas variables
	JSR MOSTRAR_PANT
	
	CMP.B #1, FLAGALM * Comprobamos si está encencida la Alarma
	BNE VUELVECLK
	CMP.B #1, FLAGALMSOUND
	BEQ ENCIENDELED * Si es la hora enncendemos leds y sonido
	
	VUELVECLK
	
	BTST.B #0, BT * Comprobamos B1
	BNE SALTO00
	JSR BUCLEB0
	JSR APAGALED * Al pulsarlo apagamos los LEDS
	
	SALTO00
	
	BTST.B #1, BT * Comprobamos B2
	BNE SETAL
	JSR BUCLEB1
	BCHG #0, FLAGALM * Al pulsarlo activamos o desactivamos la Alarma
	
	SETAL
	BTST.B #2, BT * Comprobamos B3
	BNE CLK * Si pulsamos pasamos al modo Alarma
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	
	JMP ALM * Vamos al modo alarma
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * ALARMA
	ALM
	 * Mostramos modo ALM
	MOVE.B #%00111000, MODO1 * DERECHA
	MOVE.B #%01110111, MODO2 * IZQUIERDA
	MOVE.B #%00000000, LEDS * Apagamos LEDs
	
	 * Ponemos a 0 los segundos, ya que no se tienen en cuenta en la alarma
	MOVE.B #0, PS1
	MOVE.B #0, PS2
	
	 * Pasamos los datos guardados de la alarma a los de Pantalla
	MOVE.B (MN1ALMG), PM1
	MOVE.B (MN2ALMG), PM2
	MOVE.B (HR1ALMG), PH1
	MOVE.B (HR2ALMG), PH2
	
	 * Mostramos por pantalla esas variables
	JSR MOSTRAR_PANT
	
	 * Pasamos los datos guardados al Contador
	MOVE.B (MN1ALMG), CM1
	MOVE.B (MN2ALMG), CM2
	MOVE.B (HR1ALMG), CH1
	MOVE.B (HR2ALMG), CH2
	
	
	BTST.B #1, BT * Comprobamos B2
	BNE SALTO
	JSR BUCLEB1
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP MOD_ALM * Si lo pulsamos podemos modificarla
	
	SALTO
	BTST.B #2, BT * Comprobamos B3
	BNE ALM
	JSR BUCLEB2 * Si lo pulsamos vamos al cronometro
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CRON * Modo CRON
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * MODIFICACIONES ALARMA
	MOD_ALM * Se modifican los valores de la alarma
	MOVE.B #%01000000, LEDS * Encendemos el LED de la hora
	
	BTST.B #0, BT * Comprobamos B1
	BNE CAMB_ALM_H
	JSR BUCLEB0
	JSR SUM_HOR * Si lo pulsamos sumamos 1 a las horas
	
	CAMB_ALM_H
	 * Mostramos los datos cambiados
	MOVE.B (CH1), PH1
	MOVE.B (CH2), PH2
	
	JSR MOSTRAR_PANT
	
	 * Movemos los datos cambiados a los guardados
	MOVE.B (CH1), HR1ALMG
	MOVE.B (CH2), HR2ALMG
	
	BTST.B #2, BT * Comprobamos B3
	BNE SALTOB
	JSR BUCLEB2 * Si lo pulsamos vamos al cronometro
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CRON
	
	SALTOB * CAMBIAR NOMBRE
	
	BTST.B #1, BT * Comprobamos B2
	BNE MOD_ALM
	JSR BUCLEB1
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP MOD_MIN_ALM * Si lo pulsamos podemos modificar los minutos
	
	MOD_MIN_ALM
	MOVE.B #%00010000, LEDS * Encendemos el LED de los minutos
	
	BTST.B #0, BT * Comprobamos B1
	BNE CAMB_ALM_M
	JSR BUCLEB0
	JSR SUM_MIN
	
	CAMB_ALM_M
	 * Mostramos los datos cambiados
	MOVE.B (CM1), PM1
	MOVE.B (CM2), PM2
	
	JSR MOSTRAR_PANT
	
	MOVE.B (CM1), MN1ALMG
	MOVE.B (CM2), MN2ALMG
	
	BTST.B #2, BT
	BNE SALTOB2
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CRON
	
	SALTOB2
	
	BTST.B #1, BT * Comprobamos B2
	BNE MOD_MIN_ALM
	JSR BUCLEB1
	
	JMP ALM * VOLVEMOS AL MODO ALARMA
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * CRONOMETRO
	CRON
	 * Mostramos Modo CRON
	MOVE.B #%00110111, MODO1 * DERECHA
	MOVE.B #%00111001, MODO2 * IZQUIERDA
	
	MOVE.B #%00000000, LEDS * Apagamos LEDs
	
	CMP.B #1, $8000
	BEQ SALTO1
	CMP.B #2, $8000 * CONDICION PARADO
	BEQ SALTO5
	CMP.B #3, $8000 * CONDICION FREEZE
	BEQ SALTOB3
	
	
	MOVE.B (SG1CRON), PS1
	MOVE.B (SG2CRON), PS2
	MOVE.B (MN1CRON), PM1
	MOVE.B (MN2CRON), PM2
	MOVE.B (HR1CRON), PH1
	MOVE.B (HR2CRON), PH2
	
	JSR MOSTRAR_PANT
	
	BTST.B #2, BT
	BNE SALTOO * CAMBIAR NOMBRE DEL SALTO
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP ST_START
	
	
	
	SALTOO * CAMBIAR NOMBRE DEL SALTO
	
	BTST.B #0, BT * Empezamos cronometro al pulsar B1
	BNE CRON
	JSR BUCLEB0 * Nos vamos al bucle infinito
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JSR RST_CNTCRON * PODEMOS BORRARLO
	
	 * * * * * * * * * * * * * * * * * * * * * * * 
	START_CRON
	
	MOVE.B (SG1CRON), CS1CRON
	MOVE.B (SG2CRON), CS2CRON
	MOVE.B (MN1CRON), CM1CRON
	MOVE.B (MN2CRON), CM2CRON
	MOVE.B (HR1CRON), CH1CRON
	MOVE.B (HR2CRON), CH2CRON
	
	SALTO1
	MOVE.B (CS1CRON), PS1
	MOVE.B (CS2CRON), PS2
	MOVE.B (CM1CRON), PM1
	MOVE.B (CM2CRON), PM2
	MOVE.B (CH1CRON), PH1
	MOVE.B (CH2CRON), PH2
	
	JSR MOSTRAR_PANT
	
	MOVE.B #1, $8000
	
	BTST.B #2, BT * Si pulsamos B3 vamos al modo SET
	BNE SALTO2
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP ST_START * Modo SET
	
	SALTO2
	BTST.B #1, BT * Si pulsamos B2 congelamos la pantalla y sigue contando
	BNE SALTO3
	JSR BUCLEB1
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JSR FREEZE
	
	SALTO3
	BTST.B #0, BT * Si pulsamos B1 paramos el cronometro
	BNE SALTO4
	JSR BUCLEB0
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP PARADO * Modo Parar
	
	SALTO4
	JMP SALTO1
	
	 * * * * * * * * * * * * * * * * * * * * * * * 
	PARADO
	
	MOVE.B #2, $8000
	
	MOVE.B (CS1CRON), SG1CRON
	MOVE.B (CS2CRON), SG2CRON
	MOVE.B (CM1CRON), MN1CRON
	MOVE.B (CM2CRON), MN2CRON
	MOVE.B (CH1CRON), HR1CRON
	MOVE.B (CH2CRON), HR2CRON
	
	SALTO5
	 * LO HE PUESTO PARA CUANDO DA LA VUELTA, SE PUEDE USAR UNA SUBRUTINA
	MOVE.B (SG1CRON), PS1
	MOVE.B (SG2CRON), PS2
	MOVE.B (MN1CRON), PM1
	MOVE.B (MN2CRON), PM2
	MOVE.B (HR1CRON), PH1
	MOVE.B (HR2CRON), PH2
	
	JSR MOSTRAR_PANT
	
	BTST.B #2, BT
	BNE SALTO6
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP ST_START
	
	SALTO6
	BTST.B #1, BT
	BNE SALTO7
	JSR BUCLEB1
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JSR RST_CNTCRON
	
	SALTO7
	BTST.B #0, BT
	BNE SALTO5
	JSR BUCLEB0
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP START_CRON
	
	 * * * * * * * * * * * * * * * * * * * * * * * 
	SALTOB3 * LO PUESTO PARA CUANDO DA LA VUELTA
	
	MOVE.B (SG1CRONFREEZE), PS1
	MOVE.B (SG2CRONFREEZE), PS2
	MOVE.B (MN1CRONFREEZE), PM1
	MOVE.B (MN2CRONFREEZE), PM2
	MOVE.B (HR1CRONFREEZE), PH1
	MOVE.B (HR2CRONFREEZE), PH2
	
	JSR MOSTRAR_PANT
	
	FREEZE
	
	MOVE.B (PS1), SG1CRONFREEZE
	MOVE.B (PS2), SG2CRONFREEZE
	MOVE.B (PM1), MN1CRONFREEZE
	MOVE.B (PM2), MN2CRONFREEZE
	MOVE.B (PH1), HR1CRONFREEZE
	MOVE.B (PH2), HR2CRONFREEZE
	
	MOVE.B (CS1CRON), SG1CRON
	MOVE.B (CS2CRON), SG2CRON
	MOVE.B (CM1CRON), MN1CRON
	MOVE.B (CM2CRON), MN2CRON
	MOVE.B (CH1CRON), HR1CRON
	MOVE.B (CH2CRON), HR2CRON
	
	
	MOVE.B #3, $8000
	
	BTST.B #2, BT
	BNE SALTO8
	JSR BUCLEB2
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP ST_START
	
	SALTO8
	BTST.B #1, BT
	BNE FREEZE
	JSR BUCLEB1
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP SALTO1
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * SET HORA
	
	 * * * * * ACORDARSE PONER FLAG PARA CRONO
	ST_START
	 * Pasamos los datos de la interrupcion a los modificables
	MOVE.B (MN1), CM1
	MOVE.B (MN2), CM2
	MOVE.B (HR1), CH1
	MOVE.B (HR2), CH2
	
	ST_CLK
	
	 * Mostramos Modo SET
	MOVE.B #%00111000, MODO1 * DERECHA
	MOVE.B #%01101101, MODO2 * IZQUIERDA
	
	MODIFICAMOSEGUNDOS
	
	MOVE.B #%00000100, LEDS * Encendemos el LED de la hora
	
	JSR MOVERPANTALLA
	
	BTST.B #2, BT * Comprobamos B3
	BNE SALTO9
	JSR BUCLEB2
	MOVE.B #0, LEDS
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CLK * Si lo pulsamos vamos al modo CLK
	
	SALTO9
	BTST.B #0, BT * Comprobamos B1
	BNE SALTO10
	JSR BUCLEB0
	MOVE.B #0, SG1 * Si lo pulsamos ponemos a 0 los segundos
	MOVE.B #0, SG2
	
	SALTO10
	BTST.B #1, BT
	BNE MODIFICAMOSEGUNDOS
	JSR BUCLEB1
	
	MODIFICAMOSHGORAS
	
	MOVE.B #%01000000, LEDS * Encendemos el LED de la hora
	
	JSR MOVERPANTALLA
	
	BTST.B #2, BT
	BNE SALTO11
	JSR BUCLEB2
	MOVE.B #0, LEDS
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CLK
	
	SALTO11
	BTST.B #0, BT
	BNE SALTO12
	JSR BUCLEB0
	
	JSR SUM_HOR
	MOVE.B (CH1), HR1
	MOVE.B (CH2), HR2
	
	SALTO12
	BTST.B #1, BT
	BNE MODIFICAMOSHGORAS
	JSR BUCLEB1
	
	
	MODIFICAMOSMINUTOS
	
	MOVE.B #%00010000, LEDS * Encendemos el LED de la hora
	
	JSR MOVERPANTALLA
	
	BTST.B #2, BT
	BNE SALTO13
	JSR BUCLEB2
	MOVE.B #0, LEDS * Apagamos los LEDs antes de volver al modo CLK
	
	LEA BEEP, A1 * Suena el boton
	MOVE #70, D0
	TRAP #15
	JMP CLK
	
	SALTO13
	BTST.B #0, BT
	BNE SALTO14
	JSR BUCLEB0
	JSR SUM_MIN
	MOVE.B (CM1), MN1
	MOVE.B (CM2), MN2
	
	SALTO14
	BTST.B #1, BT
	BNE MODIFICAMOSMINUTOS
	JSR BUCLEB1
	JMP MODIFICAMOSEGUNDOS
	
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * INTERRUPCIONES
	INT_CLK * CONTADOR CLK
	
	JSR INT_ALM
	
	ADD.B #1, SG1
	
	JSR CNTCRON
	
 * Definimos: SG1
	CMP.B #10, SG1
	BNE VOLVER
	SUB.B #10, SG1
	
 * Definimos: SG2
	ADD.B #1, SG2
	CMP.B #6, SG2
	BNE VOLVER
	SUB.B #6, SG2
	
 * Definimos: MN1
	ADD.B #1, MN1
	CMP.B #10, MN1
	BNE VOLVER
	SUB.B #10, MN1
	
	
 * Definimos: MN2
	ADD.B #1, MN2
	CMP.B #6, MN2
	BNE VOLVER
	SUB.B #6, MN2
	
 * Definimos: HR1
	CMP.B #2, HR2
	BEQ HOR3
	ADD.B #1, HR1
	CMP.B #10, HR1
	BNE VOLVER
	
 * Definimos: HR2
	ADD.B #1, HR2
	CMP.B #2, HR2
	BNE VOLVER
	
 * Definimos: HR1
	HOR3
	ADD.B #1, HR1
	CMP.B #4, HR1
	BNE VOLVER
	JSR RST_CLK
	VOLVER RTE
	INT_ALM
	CMP.B #1, FLAGALM * Comprobamos si está activado el modo alarma
	BNE VOLVER1 * Si esta desactivado no ejecutamos la instruccion
	
	MOVE.B HR2ALMG, D6
	MOVE.B HR2, D7
	CMP.B D6, D7
	BNE VOLVER1
	
	MOVE.B HR1ALMG, D6
	MOVE.B HR1, D7
	CMP.B D6, D7
	BNE VOLVER1
	
	MOVE.B MN2ALMG, D6
	MOVE.B MN2, D7
	CMP.B D6, D7
	BNE VOLVER1
	
	MOVE.B MN1ALMG, D6
	MOVE.B MN1, D7
	CMP.B D6, D7
	BNE VOLVER1
	
	MOVE.B SG2, D7
	CMP.B #0, D7
	BNE VOLVER1
	
	MOVE.B SG1, D7
	CMP.B #0, D7
	BNE VOLVER1
	
	MOVE.B #1, FLAGALMSOUND
	VOLVER1 RTS * Continua el programa
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * FUNCIONES
	RST_CLK * Reseta las variables del contador generico
	MOVE.B #0, SG1
	MOVE.B #0, SG2
	MOVE.B #0, MN1
	MOVE.B #0, MN2
	MOVE.B #0, HR1
	MOVE.B #0, HR2
	RTS
	
	RST_CNT * Reseta las variables del contador generico
	MOVE.B #0, CM1
	MOVE.B #0, CM2
	MOVE.B #0, CH1
	MOVE.B #0, CH2
	RTS
	RST_CNTCRON * Reseta las variables del contador del Cron
	MOVE.B #0, SG1CRON
	MOVE.B #0, SG2CRON
	MOVE.B #0, MN1CRON
	MOVE.B #0, MN2CRON
	MOVE.B #0, HR1CRON
	MOVE.B #0, HR2CRON
	
	MOVE.B (SG1CRON), PS1
	MOVE.B (SG2CRON), PS2
	MOVE.B (MN1CRON), PM1
	MOVE.B (MN2CRON), PM2
	MOVE.B (HR1CRON), PH1
	MOVE.B (HR2CRON), PH2
	
	JSR MOSTRAR_PANT
	
	RTS
	
	RST_PANT * Reseta la pantalla
	MOVE.B #0, $E0000C
	MOVE.B #0, $E0000E
	MOVE.B #0, S1
	MOVE.B #0, S2
	MOVE.B #0, M1
	MOVE.B #0, M2
	MOVE.B #0, H1
	MOVE.B #0, H2
	RTS
	DELAY * Delay de 0.5s
	MOVE.L #23, D0
	MOVE.L #25, D1
	TRAP #15
	RTS
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * CONTADORES
	SUM_MIN * SUMADOR Generico (Alarma y SET)
	
 * Definimos: CM1
	ADD.B #1, CM1
	CMP.B #10, CM1
	BNE VOLVER2
	SUB.B #10, CM1
	
 * Definimos: CM2
	ADD.B #1, CM2
	CMP.B #6, CM2
	BNE VOLVER2
	SUB.B #6, CM2
	JMP VOLVER2
	SUM_HOR * Sumamos a las horas
 * Defimimos: H1
	CMP.B #2, CH2
	BEQ C_HOR3
	ADD.B #1, CH1
	CMP.B #10, CH1
	BNE VOLVER2
	SUB.B #10, CH1
	
 * Definimos: H2
	ADD.B #1, CH2
	CMP.B #2, CH2
	BNE VOLVER2
	
 * Definimos: H1
	C_HOR3
	ADD.B #1, CH1
	CMP.B #4, CH1
	BNE VOLVER2
	BSR RST_CNT
	
	VOLVER2 RTS * Devuelve a la instruccion previa
	CNTCRON * Contador Cronometro
	
 * Definimos: CS1
	ADD.B #1, CS1CRON
	CMP.B #10, CS1CRON
	BNE VOLVER2
	SUB.B #10, CS1CRON
	
 * Definimos: CS2
	ADD.B #1, CS2CRON
	CMP.B #6, CS2CRON
	BNE VOLVER2
	SUB.B #6, CS2CRON
	
 * Definimos: CM1
	ADD.B #1, CM1CRON
	CMP.B #10, CM1CRON
	BNE VOLVER2
	SUB.B #10, CM1CRON
	
 * Definimos: CM2
	ADD.B #1, CM2CRON
	CMP.B #6, CM2CRON
	BNE VOLVER2
	SUB.B #6, CM2CRON
	
 * Defimimos: H1
	CMP.B #2, CH2CRON
	BEQ C_HOR3CRON
	ADD.B #1, CH1CRON
	CMP.B #10, CH1CRON
	BNE VOLVER2
	
 * Definimos: H2
	ADD.B #1, CH2CRON
	CMP.B #2, CH2CRON
	BNE VOLVER2
	
 * Definimos: H1
	C_HOR3CRON
	ADD.B #1, CH1CRON
	CMP.B #4, CH1CRON
	BNE VOLVER2
	JSR RST_CNTCRON
	JMP VOLVER2
	
	MOVERPANTALLA
	MOVE.B (SG1), PS1
	MOVE.B (SG2), PS2
	MOVE.B (MN1), PM1
	MOVE.B (MN2), PM2
	MOVE.B (HR1), PH1
	MOVE.B (HR2), PH2
	
	JSR MOSTRAR_PANT
	RTS
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * COMPROBACION BOTONES
	BUCLEB2 * Boton 3
	BTST #2, (BT)
	BEQ BUCLEB2
	RTS
	
	BUCLEB1 * Boton 2
	BTST #1, (BT)
	BEQ BUCLEB1
	RTS
	BUCLEB0 * Boton 2
	BTST #0, (BT)
	BEQ BUCLEB0
	RTS
	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	 * REPRESENTAR
	MOSTRAR_PANT
 * Mostramos: SG1
	MOVE.L #TABLA_P, A2
	MOVE.W #PS1, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), S1
	
	
 * Mostramos: SG2
	MOVE.L #TABLA, A2
	MOVE.W #PS2, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), S2
	
 * Mostramos: MN1
	MOVE.L #TABLA_P, A2
	MOVE.W #PM1, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), M1
	
 * Mostramos: MN2
	MOVE.L #TABLA, A2
	MOVE.W #PM2, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), M2
	
 * Mostramos: HR1
	MOVE.L #TABLA_P, A2
	MOVE.W #PH1, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), H1
	
 * Mostramos: HR2
	MOVE.L #TABLA, A2
	MOVE.W #PH2, A3
	MOVE.B (A3), D3
	ADD.L D3, A2
	MOVE.B (A2), H2
	RTS
	
	APAGALED
	MOVE.B #0, LEDS
	MOVE.B #0, FLAGALMSOUND
	RTS
	
	ENCIENDELED * Enciende y apaga los LEDS con la Alarma
	
	BCHG #0, LEDS
	BCHG #1, LEDS
	BCHG #2, LEDS
	BCHG #3, LEDS
	BCHG #4, LEDS
	BCHG #5, LEDS
	BCHG #6, LEDS
	BCHG #7, LEDS
	
	LEA ALMSOUND, A1 * Suena la alarma
	MOVE #70, D0
	TRAP #15
	JMP VUELVECLK
	
	 * FIN
	END START
